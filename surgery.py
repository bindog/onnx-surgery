import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper

class Surgery(object):
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)

    def export(self, file_name):
        onnx.save(self.model, file_name)

    def get_node_by_name(self, name):
        for node in self.model.graph.node:
            if node.name == name:
                return node

    def get_nodes_by_optype(self, typename):
        nodes = []
        for node in self.model.graph.node:
            if node.op_type == typename:
                nodes.append(node)
        return nodes

    def get_weight_by_name(self, name):
        for weight in self.model.graph.initializer:
            if weight.name == name:
                return weight

    def set_weight_by_name(self, name, data_numpy=None, all_ones=False, all_zeros=False):
        weight = self.get_weight_by_name(name)
        self.set_weight(weight, data_numpy, all_ones, all_zeros)

    def remove_node_by_name(self, name):
        target_node = self.get_node_by_name(name)
        self.remove_node(target_node)

    def remove_node(self, target_node):
        '''
            remove the node with only one input and only one output
        '''
        node_input = target_node.input[0]
        node_output = target_node.output[0]
        # set input of successor node to predecessor node of target node
        for node in self.model.graph.node:
            for i, n in enumerate(node.input):
                if n == node_output:
                    node.input[i] = node_input

        target_names = set(target_node.input) & set([weight.name for weight in self.model.graph.initializer])
        self.remove_weights(target_names)
        target_names.append(node_output)
        self.remove_inputs(target_names)
        self.remove_value_infos(target_names)
        self.model.graph.node.remove(target_node)


    def remove_weights(self, name_list):
        rm_list = []
        for weight in self.model.graph.initializer:
            if weight.name in name_list:
                rm_list.append(weight)
        for weight in rm_list:
            self.model.graph.initializer.remove(weight)


    def remove_inputs(self, name_list):
        rm_list = []
        for input_t in self.model.graph.input:
            if input_t.name in name_list:
                rm_list.append(input_t)
        for input_t in rm_list:
            self.model.graph.input.remove(input_t)

    def remove_value_infos(self, name_list):
        rm_list = []
        for value_info in self.model.graph.value_info:
            if value_info.name in name_list:
                rm_list.append(value_info)
        for value_info in rm_list:
            self.model.graph.value_info.remove(value_info)

    def set_weight(self, weight, data_numpy=None, all_ones=False, all_zeros=False):
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
                print("Warning: the new weight shape is not consistent with original shape!")
                weight.dims[:] = list(new_shape)
                for model_input in self.model.graph.input:
                    if model_input.name == weight.name:
                        # copy from onnx.helper...
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

    def set_node_attribute_by_name(self, node_name, attr_name, attr_value):
        node = self.get_node_by_name(node_name)
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
