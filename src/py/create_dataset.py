import hashlib
import subprocess

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

from tqdm import tqdm

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
    WORDS += [
        "#", "##", "###", "####", "#####",
        "?", "$", "+", "-", "/", "!", "%",
        "&", "(", ")", "*", "@", "[", "]",
        "^", "_", "~"
    ]
    WORDS += [str(i) for i in range(10000)]

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


def choose_font(font_size_range=(40, 46)) -> ImageFont.FreeTypeFont:
    fontname = _choose_font_name()
    fontsize = np.random.randint(*font_size_range)
    return ImageFont.truetype(fontname, size=fontsize)


def generate_image(height: int,
                   noise: bool = False) -> Tuple[Image.Image, str]:
    font = choose_font()
    text = choose_word()

    textsize_x, textsize_y = font.getsize(text)
    if textsize_y > height * 2:
        raise ValueError(
            f"Given height is not enough for the fontsize {font.size}")
    width = textsize_x + 15

    x = np.random.randint(0, width - textsize_x)
    y = np.random.randint(0, height * 2 - textsize_y)

    image = Image.new("RGB", (width, height * 2), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    if noise:
        img = AUGMENTOR.augment_image(np.asarray(image))
        image = Image.fromarray(img)
        image = image.resize((width // 2, height))
    return image, text


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_samples", type=int, required=True)

    args = parser.parse_args()

    output_dir = Path("data")
    images_dir = output_dir / "images"
    if args.n_samples < 1:
        raise ValueError(
            f"Number of samples must be over 1. Got {args.n_samples}")
    elif args.n_samples == 1:
        image, text = generate_image(height=32, noise=True)
        image.save(images_dir / "sample.png")
    else:
        texts = []
        names = []
        heights = []
        widths = []
        for i in tqdm(range(args.n_samples)):
            try:
                name = hashlib.md5(str(i).encode("utf-8")).hexdigest()
                image, text = generate_image(height=32, noise=True)
                width, height = image.size

                if 32 > width or width > 200:
                    continue

                if text == "":
                    continue

                names.append(name)
                texts.append(text)
                widths.append(width)
                heights.append(height)

                image.save(images_dir / f"{name}.png")
            except ValueError:
                pass

        labels = pd.DataFrame({
            "image_id": names,
            "text": texts,
            "width": widths,
            "height": heights
        })
        labels.to_csv(output_dir / "labels.csv", index=False)
