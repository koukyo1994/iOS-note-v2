import coremltools
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import utils

from PIL import Image

from dataset import CHARS

if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    model = coremltools.models.MLModel(config["mlmodel_path"], useCPUOnly=True)

    labels = pd.read_csv("data/labels.csv")
    image = cv2.imread(f"data/images/{labels.loc[0, 'image_id']}.png")

    new_image = np.ones((32, 200, 3), dtype=np.uint8) * 255
    width = image.shape[1]
    new_image[:, :width, :] = image

    pred = model.predict({"input_1": Image.fromarray(new_image)})["Identity"]
    import pdb
    pdb.set_trace()
    decoded, _ = tf.keras.backend.ctc_decode(pred[0, 0], [50])
    text = "".join((CHARS[idx - 1] for idx in decoded[0][0].numpy()))

    plt.imshow(new_image)
    plt.title(f"Answer: {labels.loc[0, 'text']}, Pred: {text}")
    plt.show()
