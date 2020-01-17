import os

import tensorflow as tf

import utils

from dataset import segmentation_dataset
from loss import segmentation_loss
from model import get_character_segmentation_model

HEIGHT = 32

if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if config.get("task") is None or config.get("task") != "segmentation":
        raise RuntimeError("Given task is not segmentation. Aborting.")

    model = get_character_segmentation_model(
        input_shape=(HEIGHT, 224, 3), filters=config["model"]["filters"])
    optimizer = tf.keras.optimizers.Adam(config["training"]["lr"])
    lossess = tf.keras.metrics.Mean(name="recognition/loss")
    dataset = segmentation_dataset(
        config["training"]["batch_size"],
        label_path=config["dataset"]["label_path"],
        image_path=config["dataset"]["image_path"],
        boxes_path=config["dataset"]["boxes_path"])

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            y_pred = model(images, training=True)
            loss = segmentation_loss(y_pred, labels)
        lossess(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def train(dataset):
        for images, labels in dataset:
            train_step(images, labels)
            if tf.equal(optimizer.iterations % 100, 0):
                tf.print("Step", optimizer.iterations, ": Loss :",
                         lossess.result())

    for epoch in range(config["training"]["n_epochs"]):
        print(f"Epoch {epoch + 1}")
        train(dataset)
        tf.print("Step", optimizer.iterations, ": Loss : ", lossess.result())

        print(f"Save weights at epoch: {epoch}")
        model.save(config["save_path"])
