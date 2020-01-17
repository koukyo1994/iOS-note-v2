import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils

from create_dataset import generate_charbox, TextBox
from model import get_character_segmentation_model


def predict(image: np.ndarray, model: tf.keras.Model) -> str:
    height = image.shape[0]
    image = cv2.resize(image, (int(image.shape[1] / height * 32), 32))
    padded = np.zeros((32, 224, 3))
    padded[:, :image.shape[1], :] = image
    input = (padded.astype(np.float32) / 127.5 - 1.0)[np.newaxis, :, :, :]
    pred = model.predict(input)
    return pred[0]


if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    model = get_character_segmentation_model(
        input_shape=(32, 224, 3), filters=config["model"]["filters"])
    model.load_weights(config["save_path"])

    image, boxes = generate_charbox(height=32, noise=False, padding=4)
    img = np.asarray(image)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for box in boxes:
        mask[box.ymin:box.ymax, box.xmin:box.xmax] = 1

    pred = predict(img, model)
    plt.imshow(pred)
    plt.savefig(f"data/prediction/pred_box.png")
    plt.imshow(img)
    plt.savefig(f"data/prediction/original.png")
    plt.imshow(mask)
    plt.title(f"n characters: {len(boxes)}")
    plt.savefig(f"data/prediction/mask.png")
