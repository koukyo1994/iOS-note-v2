import coremltools
import tensorflow as tf

import utils

from coremltools.models.neural_network import flexible_shape_utils as flex

from dataset import CHARS
from model import get_model

if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    model = get_model(
        input_shape=(32, 200, 3),
        n_vocab=len(CHARS) + 1,
        n_blocks=config["model"]["n_blocks"])
    model.load_weights(config["save_path"])

    input_name = model.inputs[0].name.split(":")[0]
    output_node_name = model.outputs[0].name.split(":")[0]
    graph_output_node_name = output_node_name.split("/")[-1]

    model.save(config["save_path"])

    coremlmodel = coremltools.converters.tensorflow.convert(
        config["save_path"],
        input_name_shape_dict={input_name: (1, 32, 200, 3)},
        output_names="output",
        image_input_names=input_name,
        image_scale=2 / 255.0,
        red_bias=-1.0,
        blue_bias=-1.0,
        green_bias=-1.0,
        is_bgr=True,
        minimum_ios_deployment_target="13")
    coremlmodel.save(config["mlmodel_path"])

    spec = coremltools.utils.load_spec(config["mlmodel_path"])

    img_size_range = flex.NeuralNetworkImageSizeRange()
    img_size_range.add_height_range((32, 32))
    img_size_range.add_width_range((32, 200))

    output_shape_range = flex.NeuralNetworkMultiArrayShapeRange()
    output_shape_range.add_channel_range((1, 1))
    output_shape_range.add_width_range((91, 91))
    output_shape_range.add_height_range((8, 50))
    flex.update_image_size_range(
        spec, feature_name=input_name, size_range=img_size_range)
    flex.update_multiarray_shape_range(
        spec, feature_name="Identity", shape_range=output_shape_range)
    coremltools.utils.save_spec(spec, config["mlmodel_path"])
