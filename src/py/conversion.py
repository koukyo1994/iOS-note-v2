import tfcoreml

import utils

from dataset import CHARS
from model import get_model

if __name__ == "__main__":
    parser = utils.get_parser()
    config = utils.load_config(parser.parse_args())

    model = get_model(
        input_shape=(32, None, 3),
        n_vocab=len(CHARS) + 1,
        n_blocks=config["model"]["n_blocks"])
    model.load_weights(config["save_path"])

    input_name = model.inputs[0].name.split(":")[0]
    output_node_name = model.outputs[0].name.split(":")[0]
    graph_output_node_name = output_node_name.split("/")[-1]

    coremlmodel = tfcoreml.convert(
        tf_model_path=config["save_path"],
        input_name_shape_dict={input_name: (32, None, 3)},
        output_feature_names=[graph_output_node_name],
        minimum_ios_deployment_target="13")
    coremlmodel.save(config["mlmodel_path"])
