from __future__ import annotations

import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .data import MNIST_MEAN, MNIST_STD
from .model import DigitCNN
from .utils import load_checkpoint, resolve_device


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def prepare_digit_image(image: Image.Image) -> Image.Image:
    """Convert a white-on-black drawing into a centered 28x28 MNIST-style image."""
    grayscale = image.convert("L")
    bbox = grayscale.point(lambda pixel: 255 if pixel > 24 else 0).getbbox()
    if bbox is None:
        return Image.new("L", (28, 28), color=0)

    cropped = grayscale.crop(bbox)
    side = max(cropped.size)
    canvas = Image.new("L", (side, side), color=0)
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    canvas.paste(cropped, offset)
    resized = canvas.resize((20, 20), Image.Resampling.LANCZOS)
    result = Image.new("L", (28, 28), color=0)
    result.paste(resized, (4, 4))
    return result


def predict_image(model: DigitCNN, image: Image.Image, device: torch.device) -> dict[str, object]:
    prepared = prepare_digit_image(image)
    tensor = image_transform()(prepared).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=1)[0]
        prediction = int(probabilities.argmax().item())

    return {
        "digit": prediction,
        "confidence": float(probabilities[prediction].item()),
        "probabilities": [float(value) for value in probabilities.cpu()],
    }


def predict(checkpoint_path: str, image_path: str, device_name: str = "auto") -> int:
    device = resolve_device(device_name)
    model = DigitCNN().to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = ImageOps.invert(Image.open(image_path).convert("L"))
    result = predict_image(model, image, device)
    prediction = int(result["digit"])

    print(f"prediction={prediction}")
    return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a digit from a local image.")
    parser.add_argument("--checkpoint", default="models/best_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to a digit image.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.checkpoint, args.image, args.device)
