import coremltools
import tensorflow as tf

import utils

from model import get_character_segmentation_model

if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    model = get_character_segmentation_model(
        input_shape=(32, 224, 3), filters=1)
    model.load_weights(config["save_path"])

    input_name = model.inputs[0].name.split(":")[0]
    output_node_name = model.outputs[0].name.split(":")[0]
    graph_output_node_name = output_node_name.split("/")[-1]

    model.save(config["save_path"])

    coremlmodel = coremltools.converters.tensorflow.convert(
        config["save_path"],
        input_name_shape_dict={input_name: (1, 32, 224, 3)},
        output_names="output",
        image_input_names=input_name,
        image_scale=2 / 255.0,
        red_bias=-1.0,
        blue_bias=-1.0,
        green_bias=-1.0,
        is_bgr=True,
        minimum_ios_deployment_target="13")

    coremlmodel.save(config["mlmodel_path"])

    spec = coremlmodel.get_spec()
    spec_fp16 = coremltools.utils.convert_neural_network_spec_weights_to_fp16(
        spec)
    coremltools.utils.save_spec(spec_fp16, config["mlmodel_path"])
