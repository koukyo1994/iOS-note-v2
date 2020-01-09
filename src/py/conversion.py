import coremltools
import tensorflow as tf

import utils

from dataset import CHARS
from model import get_model

if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    tf.compat.v1.enable_eager_execution()

    model = get_model(
        input_shape=(32, None, 3),
        n_vocab=len(CHARS) + 1,
        n_blocks=config["model"]["n_blocks"])
    model.load_weights(config["save_path"])

    input_name = model.inputs[0].name.split(":")[0]
    output_node_name = model.outputs[0].name.split(":")[0]
    graph_output_node_name = output_node_name.split("/")[-1]

    coremlmodel = coremltools.converters.tensorflow.convert(
        config["save_path"],
        input_names="image",
        output_names="output",
        image_input_names="image",
        minimum_ios_deployment_target="13")
    coremlmodel.save(config["mlmodel_path"])
