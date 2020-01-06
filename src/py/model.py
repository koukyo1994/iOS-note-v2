import tensorflow as tf
import tensorflow.keras.layers as L


def _bn_relu(x):
    x = L.BatchNormalization()(x)
    return L.ReLU()(x)


def _conv_bn_relu(x, filters, kernel_size, conv_name=None, **conv_params):
    conv_params.setdefault("strides", (1, 1))
    conv_params.setdefault("dilation_rate", (1, 1))
    conv_params.setdefault("kernel_initializer", "he_normal")
    conv_params.setdefault("padding", "same")
    conv_params.setdefault("kernel_regularizer",
                           tf.keras.regularizers.l2(1.0e-4))
    x = L.Conv2D(filters, kernel_size, name=conv_name, **conv_params)(x)
    return _bn_relu(x)


def _bn_relu_conv(x, filters, kernel_size, conv_name=None, **conv_params):
    conv_params.setdefault("strides", (1, 1))
    conv_params.setdefault("dilation_rate", (1, 1))
    conv_params.setdefault("kernel_initializer", "he_normal")
    conv_params.setdefault("padding", "same")
    conv_params.setdefault("kernel_regularizer",
                           tf.keras.regularizers.l2(1.0e-4))
    x = _bn_relu(x)
    x = L.Conv2D(filters, kernel_size, name=conv_name, **conv_params)(x)
    return x


def _shortcut(input_feature,
              residual,
              stride_width,
              stride_height,
              conv_name_base=None,
              bn_name_base=None):
    input_shape = input_feature.shape.as_list()
    residual_shape = residual.shape.as_list()

    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input_feature
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        if conv_name_base is not None:
            conv_name_base = conv_name_base + "1"
        shortcut = L.Conv2D(
            filters=residual_shape[3],
            kernel_size=(1, 1),
            strides=(stride_height, stride_width),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + "1"
        shortcut = L.BatchNormalization(name=bn_name_base)(shortcut)
    return L.add([shortcut, residual])


def _basic_block(input_feature,
                 filters,
                 stage,
                 block,
                 transition_strides=(1, 1),
                 dilation_rate=(1, 1),
                 is_first_block_of_first_layer=False):
    if block < 27:
        block = "%c" % (block + 97)
    conv_name_base = "res" + str(stage) + block + "_branch"
    x = input_feature
    if is_first_block_of_first_layer:
        x = L.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=transition_strides,
            padding="same",
            dilation_rate=dilation_rate,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=conv_name_base + "2a")(x)
    else:
        x = _bn_relu_conv(
            x,
            filters=filters,
            kernel_size=(3, 3),
            strides=transition_strides,
            dilation_rate=dilation_rate,
            conv_name=conv_name_base + "2a")
    x = _bn_relu_conv(
        x,
        filters=filters,
        kernel_size=(3, 3),
        conv_name=conv_name_base + "2b")
    return _shortcut(
        input_feature,
        x,
        stride_width=transition_strides[1],
        stride_height=transition_strides[0])


def _residual_block(x, filters, stage, blocks, is_first_layer,
                    transition_dilation_rates, transition_strides):
    for i in range(blocks):
        x = _basic_block(
            x,
            filters=filters,
            stage=stage,
            block=i,
            transition_strides=transition_strides[i],
            dilation_rate=transition_dilation_rates[i],
            is_first_block_of_first_layer=(is_first_layer and i == 0))
    return x


def get_model(input_shape, n_vocab):
    input = tf.keras.layers.Input(shape=input_shape)
    x = _conv_bn_relu(input, filters=64, kernel_size=(7, 7), strides=(2, 2))
    x = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = _residual_block(
        x,
        128,
        stage=1,
        blocks=2,
        is_first_layer=True,
        transition_strides=[(1, 1), (1, 1)],
        transition_dilation_rates=[1, 1])
    x = _residual_block(
        x,
        256,
        stage=2,
        blocks=2,
        is_first_layer=False,
        transition_strides=[(2, 1), (1, 1)],
        transition_dilation_rates=[1, 1])
    x = _residual_block(
        x,
        512,
        stage=3,
        blocks=2,
        is_first_layer=False,
        transition_strides=[(2, 1), (1, 1)],
        transition_dilation_rates=[1, 1])
    x = _residual_block(
        x,
        1024,
        stage=4,
        blocks=2,
        is_first_layer=False,
        transition_strides=[(2, 1), (1, 1)],
        transition_dilation_rates=[1, 1])
    x = L.Lambda(lambda fm: tf.squeeze(fm, axis=1))(x)
    x = L.TimeDistributed(
        L.Dense(n_vocab + 1, activation="softmax", name="softmax"))(x)
    return tf.keras.Model(input, x)