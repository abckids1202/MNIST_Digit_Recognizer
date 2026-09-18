from PIL import Image

from mnist_digit_recognition.predict import prepare_digit_image


def test_prepare_digit_image_centers_non_empty_drawing() -> None:
    image = Image.new("L", (280, 280), color=0)
    for x in range(100, 180):
        for y in range(80, 210):
            image.putpixel((x, y), 255)

    prepared = prepare_digit_image(image)

    assert prepared.size == (28, 28)
    assert prepared.getbbox() is not None
    assert 2 <= prepared.getbbox()[0] <= 6
    assert 22 <= prepared.getbbox()[2] <= 26


def test_empty_drawing_stays_empty() -> None:
    prepared = prepare_digit_image(Image.new("L", (280, 280), color=0))

    assert prepared.size == (28, 28)
    assert prepared.getbbox() is None
