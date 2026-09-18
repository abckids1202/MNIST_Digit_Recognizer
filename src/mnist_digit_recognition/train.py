from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from .config import load_config
from .data import build_train_val_loaders
from .model import DigitCNN
from .utils import ensure_dir, resolve_device, set_seed, write_json


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def train(config_path: str) -> dict[str, float | int | str]:
    config = load_config(config_path)
    set_seed(config.seed)
    device = resolve_device(config.training.device)

    train_loader, val_loader = build_train_val_loaders(config)
    model = DigitCNN(dropout=config.model.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    model_dir = ensure_dir(config.outputs.model_dir)
    report_dir = ensure_dir(config.outputs.report_dir)
    checkpoint_path = model_dir / config.outputs.checkpoint_name

    best_val_accuracy = 0.0
    history = []

    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config_path": str(Path(config_path)),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

    metrics = {
        "device": str(device),
        "epochs": config.training.epochs,
        "best_val_accuracy": best_val_accuracy,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    write_json(report_dir / "train_metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST digit classifier.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
