import torch

from mnist_digit_recognition.model import DigitCNN


def test_model_output_shape() -> None:
    model = DigitCNN()
    inputs = torch.randn(4, 1, 28, 28)
    outputs = model(inputs)

    assert outputs.shape == (4, 10)
