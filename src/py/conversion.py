import tfcoreml

from dataset import CHARS
from model import get_model

if __name__ == "__main__":
    model = get_model(input_shape=(32, None, 3), n_vocab=len(CHARS) + 1)
    model.load_weights("bin/model_weights.h5")

    input_name = model.inputs[0].name.split(":")[0]
    output_node_name = model.outputs[0].name.split(":")[0]
    graph_output_node_name = output_node_name.split("/")[-1]

    coremlmodel = tfcoreml.convert(
        tf_model_path="bin/model_weights.h5",
        input_name_shape_dict={input_name: (32, None, 3)},
        output_feature_names=[graph_output_node_name],
        minimum_ios_deployment_target="13")
    coremlmodel.save("bin/coreml_weights.mlmodel")
