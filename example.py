import argparse
from onnx import numpy_helper

from surgery import Surgery


def old_mxnet_version_example(onnxsu):
    # NOTE 1
    # in some old version mxnet model, the fix_gamma in BatchNormalization is set to True,
    # but when converting to onnx model which do NOT have the fix_gamma attribute, and the
    # gamma (named scale in onnx) parameter is not all ones, it may cause result inconsistent
    # NOTE 2
    # in some old version mxnet model, the average pooling layer has an attribute "count_include_pad"
    # but is was not set when converting to onnx model, it seems like the default value is 1
    bn_nodes = onnxsu.get_nodes_by_optype("BatchNormalization")
    for bn_node in bn_nodes:
        gamma_name = bn_node.input[1]
        onnxsu.set_weight_by_name(gamma_name, all_ones=True)
    avg_nodes = onnxsu.get_nodes_by_optype("AveragePool")
    for avg_node in avg_nodes:
        onnxsu.set_node_attribute(avg_node, "count_include_pad", 1)


def tf_without_batch_size(onnxsu, batch_size=8):
    # NOTE
    # when using tf2onnx convert the tensorflow pb model to onnx
    # the input batch_size dim is not set, we can append it
    onnxsu.list_model_inputs(2)
    # onnxsu.set_model_input_shape(name="pb_input:0", shape=(32,3,256,256))
    onnxsu.set_model_input_batch_size(batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx test")
    parser.add_argument("--input", default="", type=str, required=True)
    parser.add_argument("--output", default="", type=str, required=True)
    args = parser.parse_args()

    onnxsu = Surgery(args.input)

    old_mxnet_version_example(onnxsu)
    # tf_without_batch_size(onnxsu, 16)

    onnxsu.export(args.output)
