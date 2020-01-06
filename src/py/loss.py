import tensorflow as tf


def recognition_loss(y_pred, y_true, input_width, text_length):
    pred_length = input_width // 4
    cost = tf.keras.backend.ctc_batch_cost(y_true, y_pred,
                                           pred_length[..., tf.newaxis],
                                           text_length[..., tf.newaxis])
    return tf.reduce_mean(cost)
