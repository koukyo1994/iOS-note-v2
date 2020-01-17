import tensorflow as tf


@tf.function(
    input_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                     tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                     tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                     tf.TensorSpec(shape=(None, ), dtype=tf.int32)))
def recognition_loss(y_pred, y_true, input_width, text_length):
    pred_length = input_width // 4
    cost = tf.keras.backend.ctc_batch_cost(y_true, y_pred,
                                           pred_length[..., tf.newaxis],
                                           text_length[..., tf.newaxis])
    return tf.reduce_mean(cost)


@tf.function(
    input_signature=(tf.TensorSpec(shape=(32, 224, 3), dtype=tf.float32),
                     tf.TensorSpec(shape=(32, 224), dtype=tf.int32)))
def segmentation_loss(y_pred, y_true):
    cost = tf.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cost)
