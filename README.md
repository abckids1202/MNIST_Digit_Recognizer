# Handwritten Digit Recognition with MNIST

This project trains a simple convolutional neural network to recognize handwritten digits from the MNIST dataset.

MNIST contains:

- 60,000 training images
- 10,000 test images
- 10 digit classes: `0` through `9`
- Image size: `28 x 28` grayscale pixels

The full dataset is small. Expect about `12 MB` compressed and roughly `50 MB` after extraction/caching.

## How Much Data Do You Need?

Recommended:

- Best baseline: use the full MNIST dataset, `60,000` train and `10,000` test images.
- Good quick experiment: `10,000` train images and `2,000` test images.
- Minimum useful prototype: `1,000` to `5,000` train images, but accuracy will be less stable.

For a first real training run, use the complete dataset. MNIST is small enough that there is little reason to downsample unless your machine is very constrained.

## Project Structure

```text
mnist_digit_recognition/
  README.md
  ROADMAP.md
  requirements.txt
  app.py
  templates/
    index.html
  static/
    app.js
    styles.css
  configs/
    default.yaml
  data/
    raw/
    processed/
  models/
  reports/
  src/
    mnist_digit_recognition/
      __init__.py
      config.py
      data.py
      model.py
      train.py
      evaluate.py
      predict.py
      utils.py
  tests/
    test_model.py
```

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The project pins `numpy<2` because older CPU-only PyTorch installations cannot parse MNIST with NumPy 2.x. A current PyTorch installation can still use this same constraint.

## Dataset

By default, the code uses `torchvision.datasets.MNIST`.

If `download: true` in `configs/default.yaml`, PyTorch will download MNIST into `data/raw`.

If you want to provide the resources yourself, place the MNIST files under:

```text
data/raw/MNIST/raw/
```

Expected original files:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Compressed `.gz` versions are also commonly supported by torchvision.

## Train

```powershell
python -m mnist_digit_recognition.train --config configs/default.yaml
```

Outputs:

- Best model checkpoint: `models/best_model.pt`
- Training metrics: `reports/train_metrics.json`

## Run the Web App

After training, start the local web app:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Draw one digit on the canvas and click `Recognize digit`.

The web app uses the same preprocessing as command-line inference: it converts the drawing to grayscale, crops the non-empty pixels, centers the digit in a `28 x 28` image, and normalizes it with MNIST's mean and standard deviation.

## Evaluate

```powershell
python -m mnist_digit_recognition.evaluate --config configs/default.yaml --checkpoint models/best_model.pt
```

Outputs:

- Evaluation metrics: `reports/eval_metrics.json`

## Predict One Image

```powershell
python -m mnist_digit_recognition.predict --checkpoint models/best_model.pt --image path\to\digit.png
```

The image should contain a single handwritten digit. The prediction script converts it to grayscale, resizes it to `28 x 28`, normalizes it like MNIST, and returns the predicted class.

## Baseline Target

With the default CNN and the full MNIST dataset, a normal target is:

- `98%+` test accuracy after a few epochs
- `99%` is possible with tuning

## Architecture Summary

The model is intentionally simple:

```text
Input: 1 x 28 x 28 grayscale image
Conv2D -> ReLU -> MaxPool
Conv2D -> ReLU -> MaxPool
Flatten
Linear -> ReLU -> Dropout
Linear -> 10 class logits
```

Use this as a clean teaching/training baseline before moving to deeper architectures.

## Do You Need More Training?

You do not need additional training to support the web canvas. The full MNIST-trained CNN is enough for a first version. You should add more training data only when your real drawings differ substantially from MNIST, for example very thin strokes, unusual positioning, colored backgrounds, or multiple digits in one drawing.

The recommended next step is to collect a small set of your own labeled drawings, evaluate accuracy on that set, and fine-tune only if the web accuracy is materially lower than the MNIST test accuracy.

Current baseline measurement from this project:

- Validation accuracy: `98.85%`
- Test accuracy: `99.05%`
- Browser drawing smoke test: handwritten `7` recognized as `7` at `99.7%` confidence
