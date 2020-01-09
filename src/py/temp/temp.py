import coremltools

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as L

from coremltools.models.neural_network import flexible_shape_utils as flex

tf.compat.v1.enable_eager_execution()

input_ = L.Input(shape=(32, None, 3))
x = L.Conv2D(16, 3)(input_)
x = L.MaxPooling2D(2)(x)
model = ks.Model(input_, x)

model.summary()
model.save("temp.h5")

size_range = flex.NeuralNetworkImageSizeRange()
size_range.add_width_range((32, 200))

mlmodel = coremltools.converters.tensorflow.convert(
    "temp.h5",
    input_names="image",
    output_names="output",
    image_input_names="image",
    image_scale=1 / 255.)
mlmodel.save("temp.mlmodel")
