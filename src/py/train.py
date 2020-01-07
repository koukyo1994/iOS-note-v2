import os
import tensorflow as tf

from dataset import recognition_dataset, CHARS
from loss import recognition_loss
from model import get_model

HEIGHT = 32

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = get_model(input_shape=(HEIGHT, None, 3), n_vocab=len(CHARS) + 1)
    optimizer = tf.keras.optimizers.Adam(5e-5)
    lossess = tf.keras.metrics.Mean(name="recognition/loss")
    dataset = recognition_dataset(
        64, label_path="data/labels.csv", image_path="data/images")

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
            if tf.equal(optimizer.iterations % 10, 0):
                tf.Print("Step", optimizer.iterations, ": Loss :",
                         lossess.result())

    for epoch in range(50):
        print(f"Epoch {epoch + 1}")
        train(dataset)
        tf.Print("Step", optimizer.iterations, ": Loss : ", lossess.result())

    model.save_weights("bin/model_weights.h5")
