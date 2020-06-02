import argparse
import numpy as np

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


def tf_set_batch_size_example(onnxsu, batch_size=8):
    # NOTE
    # when using tf2onnx convert the tensorflow pb model to onnx
    # the input batch_size dim is not set, we can append it
    onnxsu.list_model_inputs(2)
    # onnxsu.set_model_input_shape(name="pb_input:0", shape=(32,3,256,256))
    onnxsu.set_model_input_batch_size(batch_size=batch_size)


def debug_internal_output(onnxsu, node_name, output_name):
    # NOTE
    # sometimes we hope to get the internal result of some node for debug,
    # but onnx do NOT have the API to support this function. Don't worry,
    # we can append an Identity OP and an extra output following the target
    # node to get the result we want
    node = onnxsu.get_node_by_name(node_name)
    onnxsu.add_extra_output(node, output_name)


def tensorrt_set_epsilon_example(onnxsu, epsilon=1e-3):
    # NOTE
    # We found when converting an onnx model with InstanceNormalization OP to TensorRT engine, the inference result is inaccurate
    # you can find the details at https://devtalk.nvidia.com/default/topic/1071094/tensorrt/inference-result-inaccurate-with-conv-and-instancenormalization-under-certain-conditions/
    # After days of debugging, and we finally find this issue is caused by the following line of code
    # https://github.com/onnx/onnx-tensorrt/blob/5dca8737851118f6ab8a33ea1f7bcb7c9f06caf5/builtin_op_importers.cpp#L1557
    # it is strange that TensorRT onnx parser only supports epsilon >= 1e-4, if you do NOT
    # want to re-compile the TensorRT OSS, you can change epsilon to 1e-3 manually...
    # I tried comment out that line, it worked but the error is bigger than setting epsilon to 1e-3
    in_nodes = onnxsu.get_nodes_by_optype("InstanceNormalization")
    for in_node in in_nodes:
        onnxsu.set_node_attribute(in_node, "epsilon", epsilon)


def add_conv_layer(onnxsu, target_node_name):
    # NOTE:
    # The name, attribute and weight of the OP can be found at:
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md
    # You must convert all your weight and attribute to the standard
    # of the ONNX to avoid unexpected error
    target_node = onnxsu.get_node_by_name(target_node_name)
    # NOTE:
    # the weight name better be complicated enough to avoid conflict,
    # And weight_dict must be in order (make sure your python version >= 3.6)
    weight_dict = {
        "W_from_a_new_conv_op": np.random.normal(0, 1, (64, 64, 3, 3)).astype(np.float32),
        "B_from_a_new_conv_op": np.random.normal(0, 1, (64,)).astype(np.float32)
    }
    attr_dict = {
        "kernel_shape": [3, 3],
        "pads": [0, 0, 0, 0]
    }
    onnxsu.insert_op_before(
                    node_name="new_conv_op",
                    target_node=target_node,
                    op_name="Conv",
                    weight_dict=weight_dict,
                    attr_dict=attr_dict
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx test")
    parser.add_argument("--input", default="", type=str, required=True)
    parser.add_argument("--output", default="", type=str, required=True)
    args = parser.parse_args()

    onnxsu = Surgery(args.input)

    # old_mxnet_version_example(onnxsu)
    # tf_set_batch_size_example(onnxsu, 16)
    # debug_internal_output(onnxsu, "your target node name", "debug_test")
    # tensorrt_set_epsilon_example(onnxsu, 1e-3)
    add_conv_layer(onnxsu, "resnetv24_batchnorm1_fwd")

    onnxsu.export(args.output)
