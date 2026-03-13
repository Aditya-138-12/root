import argparse
import hls4ml
import os
import onnx
from onnx import helper
import qonnx.util.cleanup as quc
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
# import ROOT
import traceback
import time

class Hls4ml_Parser:
    # Constructor
    def __init__(self, onnx_model_path, hls4ml_backend, only_model_configuration, only_input_output):
        try:
            self.onnx_model = onnx_model_path
            self.model = onnx.load(onnx_model_path)
            self.hls4ml_backend = hls4ml_backend
            self.only_model_configuration = only_model_configuration or False
            self.only_input_output = only_input_output or False
        except FileNotFoundError:
            print("\nError Occured: FileNotFoundError\n\nThis can generally occur if you have specified wrong path to the model, try to provide relative path to the model.\n\n")
            traceback.print_exc()
        except Exception:
            traceback.print_exc()

    # Checking the Onnx model config for smooth conversion from onnx into hls4ml in memory model object

    def check_channel_format(self) -> str:
        input_name = self.model.graph.input[0].name
        perm_shape = None
        for node in self.model.graph.node:
            if node.op_type == 'Conv' and input_name in node.input:
                print("The onnx model is in Channels First format, converting into channels last format...", end='\n')
                return True
            
            if node.op_type == 'Transpose' and input_name in node.input:
                for attr in node.attribute:
                    if attr.name == 'perm':
                        perm_shape = attr.ints
                transpose_output = node.output[0]

                for next_node in self.model.graph.node:
                    if transpose_output in next_node.input and next_node.op_type == 'Conv':
                        if list(perm_shape) == [0, 2, 3, 1]:
                            print("The onnx model Conv layer is in Channels last format, trying to instantiate hls4ml in memory model...", end='\n')
                            return False
                        elif list(perm_shape) == [0, 3, 1, 2]:
                            print("The onnx model Conv layer is in Channels first format, trying to convert it into channels last format.", end='\n') # Not tested
                            return True
        return False # No Conv layer found, edge cases should be tested (The model does not have a conv layer)

    def fix_strides(self):
        output_path = self.onnx_model
        for node in self.model.graph.node:
            if node.op_type == 'Conv':
                has_strides = any(atr.name == "strides" for atr in node.attribute)
                if not has_strides:
                    output_path = self.onnx_model.split('.')[0] + '_FixedStrides.onnx'
                    new_attr = helper.make_attribute("strides", [1, 1])
                    node.attribute.append(new_attr)
                    print('Fixed strides for node: ', node.name or 'unamed', end='\n')
                else:
                    print('strides found', end='\n')
                    return
        #onnx.save(self.model, output_path)
        self.onnx_model = output_path

    def fix_padding(self):
        output_path = self.onnx_model
        for node in self.model.graph.node:
            if node.op_type == 'Conv':
                has_padding = any(atr.name == "pads" for atr in node.attribute)
                if not has_padding:
                    output_path = self.onnx_model.split('.')[0] + '_FixedPadding.onnx'
                    new_attr = helper.make_attribute('pads', [0, 0, 0, 0])
                    node.attribute.append(new_attr)
                    print('Fixed padding for node: ', node.name or 'unamed', end='\n')
                else:
                    print('padding found', end='\n')
                    return
        #onnx.save(self.model, output_path)
        self.onnx_model = output_path

    def fix_dilations(self):
        output_path = self.onnx_model
        for node in self.model.graph.node:
            if node.op_type == 'Conv':
                has_dilations = any(atr.name == 'dilations' for atr in node.attribute)
                if not has_dilations:
                    output_path = self.onnx_model.split('.')[0] + '_FixedDilation.onnx'
                    kernel_shape = next((atr.ints for atr in node.attribute if atr.name == 'kernel_shape'), [3, 3])
                    print('Kernel Shape: ', kernel_shape)
                    dims = len(kernel_shape)
                    new_attr = helper.make_attribute("dilations", [1]*dims)
                    node.attribute.append(new_attr)
                    print('Fixed dilations for Node: ', node.name or 'unamed', end='\n')
                else:
                    print('dilations found', end='\n')
                    return
        #onnx.save(self.model, output_path)
        self.onnx_model = output_path

    def fix_group(self):
        output_path = self.onnx_model
        for node in self.model.graph.node:
            if node.op_type == 'Conv':
                has_group = any(atr.name == 'group' for atr in node.attribute)
                if not has_group:
                    output_path = self.onnx_model.split('.')[0] + '_FixedGroup.onnx'
                    new_attr = helper.make_attribute("group", 1)
                    node.attribute.append(new_attr)
                    print("Fixed group for node: ", node.name or 'unamed', end='\n')
                else:
                    print('group found', end='\n')
                    return
        onnx.save(self.model, output_path)
        self.onnx_model = output_path

    def parse_Input_layer(self, layer):
        input_layer_name = layer.attributes.get('name')
        input_layer_shape = layer.attributes.get('input_shape')
        input_result_t = layer.attributes.get('result_t').precision
        print(layer.__class__.__name__)
        print("Inputs: ", layer.inputs)
        print("Inputs Layer Name: ", input_layer_name)
        print("Input Shape: ", input_layer_shape)
        # print(f"Inputs Shape New: {layer.get_input_variable(layer.inputs[0]).shape}"), doesn't work here, cuz there can be more than one inputs.
        print("Inputs_t: ", input_result_t)
        print("Outputs: ", layer.outputs)
        print(f"Outputs Shape New: {layer.get_output_variable(layer.outputs[0]).shape}")
    
    def parse_Conv2D_layer(self, layer):
        conv2d_layer_name = layer.attributes.get('class_name')
        bias = layer.attributes.get('bias_data')
        bias_t = layer.attributes.get('bias_t').precision
        use_bias = layer.attributes.get('use_bias')
        conv_implementation = layer.attributes.get('conv_implementation')
        data_format = layer.attributes.get('data_format')
        dilation = [int(layer.attributes.get('dilation_height')), int(layer.attributes.get('dilation_width'))]
        kernel_shape = [int(layer.attributes.get('filt_height')), int(layer.attributes.get('filt_width'))]
        input_shape = [int(layer.attributes.get('in_height')), int(layer.attributes.get('in_width'))]
        channels = layer.attributes.get('n_chan')
        filters = layer.attributes.get('n_filt')
        partitions = layer.attributes.get('partitions')
        output_shape = [int(layer.attributes.get('out_height')), int(layer.attributes.get('out_width'))]
        output_t = layer.attributes.get('result_t').precision
        padding = [int(layer.attributes.get('pad_left')), int(layer.attributes.get('pad_right')), int(layer.attributes.get('pad_top')), int(layer.attributes.get('pad_bottom'))]
        strides = [int(layer.attributes.get('stride_height')), int(layer.attributes.get('stride_width'))]
        weights = layer.attributes.get('weight_data')
        weights_quantizer = layer.attributes.get('weight_quantizer')
        weights_t = layer.attributes.get('weight_t').precision
        print(layer.__class__.__name__)
        print(f"Conv2D Layer name: ", conv2d_layer_name)
        print("Inputs: ", layer.inputs)
        print("Inputs Shape: ", input_shape)
        print(f"Inputs shape new: {layer.get_input_variable(layer.inputs[0]).shape}") # a more promising way to fetch inputs, also can be used for outputs?
        print(f"Conv_Implementation: {conv_implementation}")
        print(f"Data Foramt: {data_format}\nDilation: {dilation}\nKernel Shape: {kernel_shape}")
        print(f"Channels: {channels}\nfilters: {filters}\nPartitions: {partitions}")
        print(f"Padding: {padding}\nStrides: {strides}\nWeights: {weights.shape}\nWeights Quantizer: {weights_quantizer}\nWeights_T: {weights_t}")
        print(f"Bias: {bias}\nBias_t{bias_t}\nUse_Bias: {use_bias}")
        print("Outputs: ", layer.outputs)
        print("Outputs Shape: ", output_shape)
        print(f"Output Shape New: {layer.get_output_variable(layer.outputs[0]).shape}") # yes indeed better
        print("Output_t: ", output_t)

    def parse_Conv1D_layer(self, layer):
        conv1D_layer_name = layer.attributes.get('name')
        input_shape = [int(layer.attributes.get('in_width'))]
        output_shape = [int(layer.attributes.get('out_width'))]
        channels = layer.attributes.get('n_chan')
        filters = layer.attributes.get('n_filt')
        kernel_shape = [int(layer.attributes.get('filt_width'))]
        strides = [int(layer.attribtues.get('stride_width'))]
        padding = [int(layer.attribtues.get('pad_left')), int(layer.attributes.get('pad_rigth'))]
        weights = layer.attributes.get('weight').data
        weights_t = layer.attributes.get('weight_t').precision
        bias = layer.attributes.get('bias').data
        bias_t = layer.attributes.get('bias_t').precision
        print(layer.__class__.__name__)
        print(f"Conv1D layer name: {conv1D_layer_name}")
        print(f"Inputs: {layer.inputs}")

    def parse_Transpose_layer(self, layer):
        depth = layer.attributes.get('depth')
        dim = layer.attributes.get('dim')
        height = layer.attributes.get('height')
        global_out = layer.attributes.get('global_out') or None
        perm = layer.attributes.get('perm')
        result_t = layer.attributes.get('result_t').precision
        print(layer.__class__.__name__)
        print("Inputs: ", layer.inputs)
        print(f"Inputs shape New: {layer.get_input_variable(layer.inputs[0]).shape}")
        print(f"Depth: {depth}")
        print(f"Dimension: {dim}")
        print(f"Height: {height}")
        print(f"Perm: {perm}")
        print(f"Result_t: {result_t}")
        print("Outputs: ", layer.outputs)
        print(f"Outputs Shape new: {layer.get_output_variable(layer.outputs[0]).shape}")
        # print(f"Global Out shape: {global_out.shape}\n") first check if the current layer has a global out attribute?

    def parse_Reshape_layer(self, layer):
        layer_name = layer.attributes.get('name')
        target_shape = layer.attributes.get('target_shape')
        print(layer.__class__.__name__)
        print(f"Layer Name: {layer_name}")
        print("Inputs: ", layer.inputs)
        print(f"Target SHape is: {target_shape}")
        print(f"Inputs shape: {layer.get_input_variable(layer.inputs[0]).shape}")
        print("Outputs: ", layer.outputs)
        print(f"Outputs Shape: {layer.get_output_variable(layer.outputs[0]).shape}")

    def parse_Dense_layer(self, layer):
        n_in = layer.attributes.get('n_in')
        n_out = layer.attributes.get('n_out')
        weights = layer.attributes.get('weight').data
        weights_t = layer.attributes.get('weight_t').precision
        bias = layer.attributes.get('bias').data
        bias_t = layer.attributes.get('bias_t').precision
        print(layer.__class__.__name__)
        print("Inputs: ", layer.inputs)
        print("Inputs Shape: ", n_in)
        print(f"Weights: {weights}")
        print(f"Weights_t: {weights_t}")
        print(f"Bias: {bias}")
        print(f"Bias_t: {bias_t}")
        print("Outputs: ", layer.outputs)
        print(f"Outputs Shape: {n_out}")

    def parse_Activation_layer(self, layer):
        layer_name = layer.attributes.get('name')
        activation_function = layer.attributes.get('activation')
        n_in = layer.attributes.get('n_in')
        result_t = layer.attributes.get('result_t').precision
        print(layer.__class__.__name__)
        print("Inputs: ", layer.inputs)
        print(f"Layer Input Shape: {n_in}")
        print(f"Layer Name: {layer_name}")
        print(f"Layer Activation: {activation_function}")
        print("Outputs: ", layer.outputs)
        print(f"Output_t: {result_t}")

    def parse_Concatenate_layer(self, layer):
        print(layer.__class__.__name__)
        print("Input 1: ", layer.inputs[0])
        print("Input 1 Shape: ", layer.get_input_variable(layer.inputs[0]).shape)
        print("Input 2: ", layer.inputs[1])
        print("Input 2 Shape: ", layer.get_input_variable(layer.inputs[1]).shape)
        print("Outputs: ", layer.outputs)
        print("Outputs Shape: ", layer.get_output_variable(layer.outputs[0]).shape)

    def convert_model_into_channels_last_format(self):
        try: 
            self.fix_strides()
            self.fix_padding()
            self.fix_dilations()
            self.fix_group()

            output_path = self.onnx_model.split('.')[0] + "_ChannelLastFormat.onnx"

            qonnx_model = ModelWrapper(self.model)
            qonnx_model = quc.cleanup_model(qonnx_model)
            qonnx_model = qonnx_model.transform(ConvertToChannelsLastAndClean())
            qonnx_model = quc.cleanup_model(qonnx_model)

            qonnx_model.save(output_path)
            self.onnx_model = output_path
            print('Converted the model into channels last format...', end='\n')
            return True
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()
            return False


    def instantiate_hls4ml_model_and_parse(self):
        try:

            self.model = onnx.load(self.onnx_model)

            config = hls4ml.utils.config_from_onnx_model(
                self.model,
                granularity='name',
                default_precision='fixed<16, 6>',
                backend=self.hls4ml_backend
            )

        
            hls4ml_model = hls4ml.converters.convert_from_onnx_model(
                self.model,
                hls_config=config,
                output_dir='tdeout',
                io_type='io_parallel',
                backend=self.hls4ml_backend
            )

            print("\n\nParsing the Model, getting configs, model graph....")
            
            if self.only_model_configuration:
                time.sleep(2)
                os.system('clear')

            hls4ml_backend = hls4ml_model.config.backend.__class__.__name__
            print(f"The backend of hls4ml model is: {hls4ml_backend}")
            # Naming the layers accoordig to there backend
            input = self.hls4ml_backend + 'Input'
            conv2D = self.hls4ml_backend + 'Conv2D'
            conv1D = self.hls4ml_backend + 'Conv1D'
            transpose = self.hls4ml_backend + 'Transpose'
            dense = self.hls4ml_backend + 'Dense'
            activation = self.hls4ml_backend + 'Activation'
            reshape = self.hls4ml_backend + 'Reshape'
            concatenate = self.hls4ml_backend + 'Concatenate'

            #layers = hls4ml_model.get_layers()
            # Parsing the model, supports - Dense, Transpose, Activation, Conv2d, Conv1D, Reshape, Concatenate layers at the moment
            # This function can Instantiate a Sofie RModel object for these operators at the moment, - ReLu, Gemm, Reshape, Concat 
            if not self.only_input_output:
                for layer in hls4ml_model.get_layers():
                    print("\n\n")
                    if layer.attributes.get('class_name') == 'InputLayer' or layer.__class__.__name__ == input:
                        self.parse_Input_layer(layer)
                    if layer.__class__.__name__ == conv2D:
                        self.parse_Conv2D_layer(layer)
                    if layer.__class__.__name__ == conv1D:
                        self.parse_Conv1D_layer(layer)
                    if layer.__class__.__name__ == transpose:
                        self.parse_Transpose_layer(layer)
                    if layer.__class__.__name__ == reshape:
                        self.parse_Reshape_layer(layer)
                    if layer.__class__.__name__ == dense:
                        self.parse_Dense_layer(layer)
                    if layer.__class__.__name__ == activation:
                        self.parse_Activation_layer(layer)
                    if layer.__class__.__name__ == concatenate: # At the moment, hls4ml supports 2 inputs in the concatenate layer, : https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/model/layers.py#L1184
                        self.parse_Concatenate_layer(layer)
            elif self.only_input_output:
                self.parse_Input_layer(next(layer for layer in hls4ml_model.get_layers() if layer.__class__.__name__ == self.hls4ml_backend + 'Input'))
                print("\n\n")
                output = next(layer for layer in hls4ml_model.get_layers() if layer.attributes.get('global_out'))
                fn_call = 'parse_' + output.__class__.__name__.replace(self.hls4ml_backend, '') + '_layer'
                getattr(self, fn_call)(output)
        except AttributeError:
            print("\nError occured: AttributeError\n\nThis can generally occur if you forgot to add the --backend, or the Output layer is not yet supported by the parser :().\n\n")
            traceback.print_exc()
        except StopIteration:
            print("\nError occured: StopIteration\n\nThis error occurs generally due to these reasons: \n1. You forgot to run shape inference usin onnx.shape_inference, \n2. The onnx graph consists of unsupported layers like unsqueeze, etc.\n\n")
            traceback.print_exc()
        except UnboundLocalError:
            print("\nError occured: UnboundLocalError, check logs below.\n\n")
            traceback.print_exc()
        except Exception:
            traceback.print_exc()
        return

    def __call__(self):
        ctrl = self.check_channel_format()
        if ctrl:
            nextCtrl = self.convert_model_into_channels_last_format() # Error when passing channels last format
            if nextCtrl:
                self.instantiate_hls4ml_model_and_parse()
            else:
                print("Unknown eror occured")
        if not ctrl:
            # Required to cleanup models when they donot have conv layers.
            output_path = self.onnx_model
            qonnx_model = ModelWrapper(self.model)
            qonnx_model = quc.cleanup_model(qonnx_model)
            qonnx_model.save(output_path)
            self.instantiate_hls4ml_model_and_parse()


def main():

    parser = argparse.ArgumentParser(description='CLI arguments parser')
    parser.add_argument('--modelPath', type=str, help='Onnx model path')
    parser.add_argument('--backend', type=str, help='Backend to be used while instantiating a hls4ml in memory object, Available values: Vivado, Vitis, Quartus, Catapult, oneAPI')
    parser.add_argument('-L', action='store_true', help='Prints only the HLS4ML in memory model configuration parsed by the parser')
    parser.add_argument('-IO', action='store_true', help='Prints only the input and output layer of the HLS4ML in memory model')
    args = parser.parse_args()
    onnx_model_path = args.modelPath
    hls4ml_backend = args.backend
    only_model_configuration = args.L
    only_input_output = args.IO

    hls_parser = Hls4ml_Parser(onnx_model_path, hls4ml_backend, only_model_configuration, only_input_output)
    hls_parser()

if __name__ == '__main__':
    main()