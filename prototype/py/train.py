import os

import tensorflow as tf

import utils

from dataset import recognition_dataset, CHARS
from loss import recognition_loss
from model import get_model

HEIGHT = 32

if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = get_model(
        input_shape=(HEIGHT, 200, 3),
        n_vocab=len(CHARS) + 1,
        n_blocks=config["model"]["n_blocks"])
    optimizer = tf.keras.optimizers.Adam(config["training"]["lr"])
    lossess = tf.keras.metrics.Mean(name="recognition/loss")
    eval_losses = tf.keras.metrics.Mean(name="recognition/eval_loss")
    train_dataset, test_dataset = recognition_dataset(
        config["training"]["batch_size"],
        label_path=config["dataset"]["label_path"],
        image_path=config["dataset"]["image_path"])

    # @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            y_pred = model(images, training=True)
            loss = recognition_loss(y_pred, labels["text"], labels["width"],
                                    labels["text_length"])
        lossess(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def train(dataset):
        for images, labels in dataset:
            train_step(images, labels)
            if tf.equal(optimizer.iterations % 100, 0):
                tf.print("Step", optimizer.iterations, ": Loss :",
                         lossess.result())

    def evaluate(dataset):
        for images, labels in dataset:
            y_pred = model(images, training=False)
            loss = recognition_loss(y_pred, labels["text"], labels["width"],
                                    labels["text_length"])
            eval_losses(loss)

    for epoch in range(config["training"]["n_epochs"]):
        print(f"Epoch {epoch + 1}")
        train(train_dataset)
        tf.print("Step", optimizer.iterations, ": Loss : ", lossess.result())

        evaluate(test_dataset)
        tf.print("eval loss: ", eval_losses.result())

        print(f"Save weights at epoch: {epoch}")
        model.save(config["save_path"])
