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
    print("="*10, "attributes of node: ", node.name, "="*10)
    for input_name in node.input:
        print(input_name)
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


def set_weight(weight, data_numpy=None, all_ones=False, all_zeros=False):
    # NOTE: weight can be stroed in human readable fields(float_data, int32_data, string_data, ...)
    # as well as raw_data, if we set weight by raw_data, we must clear the fields above to make it effective
    if data_numpy is not None:
        raw_shape = weight.dims
        new_shape = np.shape(data_numpy)
        if new_shape != raw_shape:
            print("Warning: the new weight shape is not consistent with original shape, it may cause error!")
        weight.ClearField("float_data")
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
        weight.raw_data = wn.tobytes()


# TODO elementwise op (Div Sub ...) constant


if __name__ == "__main__":
    test_onnx = "raw.onnx"
    out_onnx = "new.onnx"
    model = onnx.load(test_onnx)
    xx = get_nodes_by_optype(model, "Conv")
    show_node_inputs(xx[10])
    yy = get_weight_by_name(model, "conv_3b_1x1_weight")
    show_weight(yy)
    set_weight(yy, all_ones=True)
    onnx.save(model, out_onnx)
