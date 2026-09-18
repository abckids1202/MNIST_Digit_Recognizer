from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Keep `python app.py` runnable directly from this project folder.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mnist_digit_recognition.config import load_config
from mnist_digit_recognition.model import DigitCNN
from mnist_digit_recognition.predict import predict_image
from mnist_digit_recognition.utils import load_checkpoint, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pt"
config = load_config(CONFIG_PATH)
device = resolve_device(os.getenv("MNIST_DEVICE", config.training.device))

app = Flask(__name__)
model: DigitCNN | None = None
model_error: str | None = None


def load_model() -> None:
    global model, model_error
    if not CHECKPOINT_PATH.exists():
        model_error = "Model checkpoint is missing. Run the training command first."
        return

    try:
        loaded_model = DigitCNN(dropout=config.model.dropout).to(device)
        checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()
        model = loaded_model
        model_error = None
    except Exception as exc:  # pragma: no cover - surfaced through the health endpoint
        model_error = f"Could not load model checkpoint: {exc}"


load_model()


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_ready=model is not None,
        model_error=model_error,
        device=str(device),
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if model is not None else "model_unavailable",
            "model_ready": model is not None,
            "checkpoint": str(CHECKPOINT_PATH),
            "device": str(device),
            "error": model_error,
        }
    ), (200 if model is not None else 503)


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": model_error or "Model is unavailable."}), 503

    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not isinstance(data_url, str) or "," not in data_url:
        return jsonify({"error": "Send a canvas image as a data URL in the 'image' field."}), 400

    try:
        encoded_image = data_url.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(encoded_image))).convert("L")
        result = predict_image(model, image, device)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Could not read the drawing: {exc}"}), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
