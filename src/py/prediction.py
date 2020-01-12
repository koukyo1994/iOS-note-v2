import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils

from create_dataset import generate_image
from dataset import CHARS
from model import get_model


def predict_text(image: np.ndarray, model: tf.keras.Model) -> str:
    height = image.shape[0]
    image = cv2.resize(image, (int(image.shape[1] / height * 32), 32))
    padded = np.zeros(32, 200, 3)
    padded[:, :image.shape[1], :] = image
    input = (padded.astype(np.float32) / 127.5 - 1.0)[np.newaxis, :, :, :]
    pred = model.predict(input)
    decoded, _ = tf.keras.backend.ctc_decode(pred, [input.shape[2] // 4])
    return "".join((CHARS[idx - 1] for idx in decoded[0][0].numpy()))


if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    model = get_model(
        input_shape=(32, 200, 3),
        n_vocab=len(CHARS) + 1,
        n_blocks=config["model"]["n_blocks"])
    model.load_weights(config["save_path"])

    image, text = generate_image(height=32, noise=False)
    img = np.asarray(image)

    pred = predict_text(img, model)
    plt.imshow(img)
    plt.title(f"Answer: {text}, Pred: {pred}")
    plt.savefig(f"data/prediction/pred_{text}.png")
