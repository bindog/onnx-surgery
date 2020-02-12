import argparse
from onnx import numpy_helper

from surgery import Surgery


def show_node_attributes(node):
    print("="*10, "attributes of node: ", node.name, "="*10)
    for attr in node.attribute:
        print(attr.name)
    print("="*60)


def show_node_inputs(node):
    # Generally, the first input is the truely input
    # and the rest input is weight initializer
    print("="*10, "inputs of node: ", node.name, "="*10)
    for input_name in node.input:
        print(input_name)  # type of input_name is str
    print("="*60)


def show_node_outputs(node):
    # Generally, the first input is the truely input
    # and the rest input is weight initializer
    print("="*10, "outputs of node: ", node.name, "="*10)
    for output_name in node.output:
        print(output_name)  # type of output_name is str
    print("="*60)


def show_weight(weight):
    print("="*10, "details of weight: ", weight.name, "="*10)
    print("data type: ", weight.data_type)
    print("shape: ", weight.dims)
    data_numpy = numpy_helper.to_array(weight)
    # data_numpy = np.frombuffer(weight.raw_data, dtype=xxx)
    # print("detail data:", data_numpy)
    print("="*40)


# TODO elementwise op (Div Sub ...) constant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx test")
    parser.add_argument("--input", default="", type=str, required=True)
    parser.add_argument("--output", default="", type=str, required=True)
    args = parser.parse_args()

    onnxsu = Surgery(args.input)
    onnxsu.set_weight_by_name("model_main.model.layer1.0.conv1.weight", all_zeros=True)
    xx = onnxsu.get_nodes_by_optype("BatchNormalization")
    onnxsu.set_node_attribute(xx[0], "epsilon", 1e-3)
    onnxsu.export(args.output)
