import hashlib
import pickle
import subprocess

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, NamedTuple, List

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
        "#", "##", "###", "####", "#####", "?", "$", "+", "-", "/", "!", "%",
        "&", "(", ")", "*", "@", "[", "]", "^", "_", "~"
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


class TextBox(NamedTuple):
    text: str
    xmin: int
    xmax: int
    ymin: int
    ymax: int


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


def generate_charbox(height: int, padding: int = 2,
                     noise: bool = False) -> Tuple[Image.Image, List[TextBox]]:
    font = choose_font()
    text = choose_word()

    char_widths = []
    char_heights = []
    max_char_height = 0
    for char in text:
        char_width, char_height = font.getsize(char)
        char_widths.append(char_width)
        char_heights.append(char_height)
        if char_height > max_char_height:
            max_char_height = char_height
        if max_char_height > height * 2:
            raise ValueError(
                f"Given height is not enough for the fontsize {font.size}")
    width = sum(char_widths) + padding * len(char_widths)
    x = np.random.randint(0, 10)
    y = np.random.randint(0, height * 2 - max_char_height)

    image = Image.new("RGB", (width, height * 2), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    char_boxes: List[TextBox] = []
    for i, char in enumerate(text):
        draw.text((x, y), char, fill=(0, 0, 0), font=font)
        char_boxes.append(
            TextBox(
                text=char,
                xmin=x,
                xmax=x + char_widths[i],
                ymin=y,
                ymax=y + char_heights[i]))
        x = x + char_widths[i] + np.random.randint(0, padding)
        y = np.random.randint(0, height * 2 - max_char_height)

    if noise:
        img = AUGMENTOR.augment_image(np.asarray(image))
        image = Image.fromarray(img)
    image = image.resize((width // 2, height))
    return image, char_boxes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--mode", default="text")

    args = parser.parse_args()

    output_dir = Path("data")
    if args.mode == "text":
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
                    image, text = generate_image(height=32, noise=False)
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
    elif args.mode == "char_box":
        images_dir = output_dir / "char_images"
        box_dir = output_dir / "char_boxes"
        images_dir.mkdir(parents=True, exist_ok=True)
        box_dir.mkdir(parents=True, exist_ok=True)
        if args.n_samples < 1:
            raise ValueError(
                f"Number of samples must be over 1. Got {args.n_samples}")
        elif args.n_samples == 1:
            image, boxes = generate_charbox(height=32, noise=True)
            image.save(images_dir / "sample.png")
            with open(box_dir / "sample.pkl", "wb") as bf:
                pickle.dump(boxes, bf)
        else:
            names = []
            heights = []
            widths = []
            for i in tqdm(range(args.n_samples)):
                try:
                    name = hashlib.md5(str(i).encode("utf-8")).hexdigest()
                    image, boxes = generate_charbox(height=32, noise=False)
                    width, height = image.size

                    if 32 > width or width > 200:
                        continue

                    if len(boxes) == 0:
                        continue

                    names.append(name)
                    widths.append(width)
                    heights.append(height)

                    image.save(images_dir / f"{name}.png")
                    with open(box_dir / f"{name}.pkl", "wb") as bf:
                        pickle.dump(boxes, bf)
                except ValueError:
                    pass

            labels = pd.DataFrame({
                "image_id": names,
                "width": widths,
                "height": heights
            })
            labels.to_csv(output_dir / "labels_charbox.csv", index=False)
