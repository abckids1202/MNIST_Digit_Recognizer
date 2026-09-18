# Model Card: MNIST Digit Classifier

## Model

Small convolutional neural network for handwritten digit classification.

## Intended Use

Educational and prototyping use for recognizing isolated handwritten digits from MNIST-like images.

## Not Intended For

- Reading full documents.
- Recognizing letters or symbols.
- Processing photographs with complex backgrounds.
- High-stakes identity, finance, medical, or legal decisions.

## Data

MNIST handwritten digit dataset:

- `60,000` training images
- `10,000` test images
- Digits `0` through `9`
- Grayscale, `28 x 28`

## Expected Performance

With full MNIST and default hyperparameters, expected test accuracy is usually above `98%`.

## Limitations

The model is trained on centered, normalized digit images. It may perform poorly on:

- Multiple digits in one image.
- Cropped or off-center digits.
- Colored backgrounds.
- Digits written in styles unlike MNIST.

## Ethical Notes

MNIST is a low-risk educational dataset, but this model should not be treated as a general handwriting recognition system.
