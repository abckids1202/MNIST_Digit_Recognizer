from __future__ import annotations

import argparse

import torch
from torch import nn

from .config import load_config
from .data import build_test_loader
from .model import DigitCNN
from .utils import load_checkpoint, resolve_device, write_json


def evaluate(config_path: str, checkpoint_path: str) -> dict[str, float | str]:
    config = load_config(config_path)
    device = resolve_device(config.training.device)

    model = DigitCNN(dropout=config.model.dropout).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = build_test_loader(config)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

            for label, prediction in zip(labels.cpu(), predictions.cpu()):
                confusion[label, prediction] += 1

    metrics = {
        "checkpoint": checkpoint_path,
        "test_loss": total_loss / total,
        "test_accuracy": correct / total,
        "confusion_matrix": confusion.tolist(),
    }
    write_json(f"{config.outputs.report_dir}/eval_metrics.json", metrics)
    print(f"test_loss={metrics['test_loss']:.4f} test_acc={metrics['test_accuracy']:.4f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MNIST digit classifier.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", default="models/best_model.pt", help="Path to model checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.checkpoint)
