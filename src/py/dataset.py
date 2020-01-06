import subprocess

import imgaug.augmenters as iaa
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

FONTNAMES = subprocess.run(
    "fc-list :lang=en | sed -r -e 's/^(.+): .*$/\\1/g'",
    stdout=subprocess.PIPE,
    shell=True).stdout.decode("utf-8").strip().split("\n")
CUSTOMFONTS = subprocess.run(
    "fc-list | sed -r -e 's/^(.+): .*$/\\1/g' | grep custom",
    stdout=subprocess.PIPE,
    shell=True).stdout.decode("utf-8").strip().split("\n")

FONTNAMES += CUSTOMFONTS

with open("/usr/share/dict/words") as f:
    WORDS = f.read().splitlines()

AUGMENTOR = iaa.Sequential([
    iaa.OneOf([iaa.GaussianBlur(
        (0, 1.0)), iaa.AverageBlur(k=(1, 2))]),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    iaa.Dropout((0.01, 0.03), per_channel=0.1),
    iaa.Add((-10, 10), per_channel=0.5),
    iaa.OneOf([
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.FrequencyNoiseAlpha(
            exponent=(-4, 0),
            first=iaa.Multiply((0.5, 1.5), per_channel=0.5),
            second=iaa.ContrastNormalization((0.5, 2.0)))
    ]),
],
                           random_order=True)


def _choose_font_name() -> str:
    return np.random.choice(FONTNAMES)


def choose_word() -> str:
    return np.random.choice(WORDS)


def choose_font(font_size_range=(15, 30)) -> ImageFont.FreeTypeFont:
    fontname = _choose_font_name()
    fontsize = np.random.randint(*font_size_range)
    return ImageFont.truetype(fontname, size=fontsize)


def generate_image(height: int,
                   noise: bool = False) -> Tuple[Image.Image, str]:
    font = choose_font()
    text = choose_word()

    textsize_x, textsize_y = font.getsize(text)
    if textsize_y > height:
        raise ValueError(
            f"Given height is not enough for the fontsize {font.size}")
    width = textsize_x + 15

    x = np.random.randint(0, width - textsize_x)
    y = np.random.randint(0, height - textsize_y)

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    if noise:
        img = AUGMENTOR.augment_image(np.asarray(image))
        image = Image.fromarray(img)
    return image, text


if __name__ == "__main__":
    image, text = generate_image(height=32, noise=True)
    image.save("images/sample.png")
