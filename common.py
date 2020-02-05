import argparse
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper


def get_node_by_name(model, name):
    for node in model.graph.node:
        if node.name == name:
            return node


def get_nodes_by_optype(model, typename):
    nodes = []
    for node in model.graph.node:
        if node.op_type == typename:
            nodes.append(node)
    return nodes


def remove_node(model, target_node):
    # only one input and only one output
    # and generally, the output name and node name are the same
    node_input = target_node.input[0]

    # predecessor_node = get_node_by_name(model, node_input)
    # node_output = target_node.output[0]

    # set input of successor node to predecessor node of target node
    # successor_node_list = []
    for node in model.graph.node:
        for i, n in enumerate(node.input):
            # successor_node_list.append(node)
            if n == target_node.name:
                node.input[i] = node_input

    weights = model.graph.initializer
    target_names = set(target_node.input) & set([weight.name for weight in weights])
    weights_to_be_removed = [weight for weight in weights if weight.name in target_names]
    inputs_to_be_removed = [model_input for model_input in model.graph.input if model_input.name in target_names]
    for weight in weights_to_be_removed:
        print("removing weight initializer: ", weight.name)
        weights.remove(weight)
    for model_input in inputs_to_be_removed:
        print("removing weight in inputs: ", model_input.name)
        model.graph.input.remove(model_input)
    model.graph.node.remove(target_node)


def insert_flatten_before(model, target_node):
    from onnx import TensorProto
    print("test inserting a node...")
    # get target_node inputs
    node_input = target_node.input[0]
    # create new node
    node_name = "flatten_test"
    flatten_output = helper.make_tensor_value_info(node_name, TensorProto.FLOAT, [8, -1])
    flatten_node = helper.make_node('Flatten', [node_input], [node_name], name=node_name)
    # set target_node inputs to target_node outputs
    for node in model.graph.node:
        for i, n in enumerate(node.input):
            # successor_node_list.append(node)
            if n == target_node.name:
                node.input[i] = node_name
    # append value proto to model graph input
    model.graph.input.append(flatten_output)
    print("test insert node end...")


def get_weight_by_name(model, name):
    weights = model.graph.initializer
    for weight in weights:
        if weight.name == name:
            return weight


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


def set_node_attribute(node, attr_name, attr_value):
    flag = False
    for attr in node.attribute:
        if (attr.name == attr_name):
            if attr.type == 1:
                attr.f = attr_value
            elif attr.type == 2:
                attr.i = attr_value
            elif attr.type == 3:
                attr.s = attr_value
            elif attr.type == 4:
                attr.t = attr_value
            elif attr.type == 5:
                attr.g = attr_value
            # NOTE: For repeated composite types, we should use something like
            # del attr.xxx[:]
            # attr.xxx.extend([n1, n2, n3])
            elif attr.type == 6:
                attr.floats[:] = attr_value
            elif attr.type == 7:
                attr.ints[:] = attr_value
            elif attr.type == 8:
                attr.strings[:] = attr_value
            else:
                print("unsupported attribute data type right now...")
                return False
            flag = True
    return flag


def show_weight(weight):
    print("="*10, "details of weight: ", weight.name, "="*10)
    print("data type: ", weight.data_type)
    print("shape: ", weight.dims)
    data_numpy = numpy_helper.to_array(weight)
    # data_numpy = np.frombuffer(weight.raw_data, dtype=xxx)
    # print("detail data:", data_numpy)
    print("="*40)


def set_weight(model, weight, data_numpy=None, all_ones=False, all_zeros=False):
    # NOTE: weight can be stroed in human readable fields(float_data, int32_data, string_data, ...)
    # as well as raw_data, if we set weight by raw_data, we must clear the fields above to make it effective
    # NOTE: data_type between numpy and TensorProto
    if data_numpy is not None:
        raw_shape = tuple([i for i in weight.dims])
        new_shape = np.shape(data_numpy)
        if weight.data_type == 8:
            # string data type is special, it requires to store data in string_data field
            # NOT the raw_data field
            print("Can NOT handle string data type right now...")
            exit()
            # weight.string_data = bytes(data_numpy, encoding = "utf8")
            # weight.ClearField("raw_data")
        if new_shape != raw_shape:
            print("Warning: the new weight shape is not consistent with original shape, it may cause error!")
            # new_weight = numpy_helper.from_array(data_numpy, weight.name)
            # if new_weight.data_type != weight.data_type:
            #     print("Warning: the new weight data type is not consistent with original data type, it may cause error!")
            weight.dims[:] = list(new_shape)
            for model_input in model.graph.input:
                if model_input.name == weight.name:
                    # print(model_input.name)
                    # print(model_input.type.tensor_type.shape.dim)

                    # new_input = helper.make_tensor_value_info(model_input.name, onnx.TensorProto.FLOAT, new_shape)
                    # model.graph.input.remove(model_input)
                    # model.graph.input.append(new_input)

                    tensor_shape_proto = model_input.type.tensor_type.shape
                    tensor_shape_proto.ClearField("dim")
                    tensor_shape_proto.dim.extend([])
                    for d in new_shape:
                        dim = tensor_shape_proto.dim.add()
                        dim.dim_value = d

        weight.ClearField("float_data")
        weight.ClearField("int32_data")
        weight.ClearField("int64_data")
        weight.raw_data = data_numpy.tobytes()
    else:
        if all_ones:
            wr = numpy_helper.to_array(weight)
            wn = np.ones_like(wr)
        elif all_zeros:
            wr = numpy_helper.to_array(weight)
            wn = np.zeros_like(wr)
        else:
            print("You must give a data_numpy to set the weight, or set the all_ones/all_zeros flag.")
            exit()
        weight.ClearField("float_data")
        weight.ClearField("int32_data")
        weight.ClearField("int64_data")
        weight.raw_data = wn.tobytes()


# TODO elementwise op (Div Sub ...) constant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx test")
    parser.add_argument("--input", default="", type=str, required=True)
    parser.add_argument("--output", default="", type=str, required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)

    xx = get_nodes_by_optype(model, "Conv")
    show_node_inputs(xx[10])
    insert_flatten_before(model, xx[10])

    yy = get_weight_by_name(model, "conv_3b_1x1_weight")
    show_weight(yy)
    test_numpy = np.zeros((64, 256, 2, 2), dtype=np.float32)
    set_weight(model, yy, test_numpy)

    # dd = get_node_by_name(model, "bn_conv1")
    # remove_node(model, dd)

    onnx.save(model, args.output)
