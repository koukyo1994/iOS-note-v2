import cv2
import pandas as pd
import tensorflow as tf

from pathlib import Path

CHARS = "abcdefghijklmnopqrstuvwxyz"
CHARS += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS += "0123456789"
CHARS += "!#$%&()*+-./:<=>?@[\]^_{}~'"


def _get_data_generator(label_path: str, image_path: str):
    labels = pd.read_csv(label_path)
    labels.dropna(how="any", axis=0, inplace=True)

    image_path_ = Path(image_path)

    characters = list(CHARS)
    char_map = {c: i + 1 for i, c in enumerate(characters)}

    def generator():
        for _, row in labels.iterrows():
            image_id = row.image_id
            text = [char_map.get(c, 0) for c in row.text]
            width = row.width

            path_str = str(image_path_ / f"{image_id}.png")
            img = cv2.imread(path_str)
            yield (img, {
                "width": width,
                "text": text,
                "text_length": len(text)
            })

    return generator


def recognition_dataset(batch_size,
                        label_path="data/labels.csv",
                        image_path="data/images",
                        training=False):
    dataset = tf.data.Dataset.from_generator(
        _get_data_generator(label_path, image_path),
        output_types=(tf.uint8, {
            "width": tf.int32,
            "text": tf.int32,
            "text_length": tf.int32
        }))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.shuffle()
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=((32, None, 3), {
            "width": (),
            "text": (None, ),
            "text_length": ()
        }))

    def map_fn(image, labels):
        return tf.cast(image, tf.float32) / 127.5 - 1, labels

    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    dataset = recognition_dataset(
        16, label_path="data/labels.csv", image_path="data/images")
    for images, labels in dataset:
        print(images.shape)
        print(labels)
        break
