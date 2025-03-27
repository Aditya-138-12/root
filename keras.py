import hls4ml
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

# chek if we’ve got QONNX 
try:
    from qonnx.util.to_channels_last import to_channels_last
    from qonnx.core.modelwrapper import ModelWrapper
    qonnx_ready = True
except ImportError:
    qonnx_ready = False
    print("No QONNX found, we’ll do this the hard way.")

# quick check to see for the ONNX 
def check_model(onnx_file):
    try:
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model) 
        shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        looks_good = len(shape) == 4  # 4D for this to work
        is_nhwc = looks_good and shape[-1] < shape[2] and shape[-1] < shape[1]  # Channels last?
        return looks_good, is_nhwc, shape
    except Exception as oops:
        print(f"Somthing’s wrong with the model: {oops}")
        return False, False, None

# flip NCHW to NHWC if QONNX not present
def flip_to_nhwc_the_hard_way(input_file, output_file):
    model = onnx.load(input_file)
    graph = model.graph
    
    # get the input shape
    old_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    if len(old_shape) != 4:
        raise ValueError(f"Whoa, expected a 4D shape, got {old_shape} insted.")
    
    # rearrange to NHWC: batch, height, width, chanels
    new_shape = [old_shape[0], old_shape[2], old_shape[3], old_shape[1]]
    input_tensor = graph.input[0]
    input_tensor.type.tensor_type.shape.Clear()  # Change the old shape?
    for size in new_shape:
        dim = input_tensor.type.tensor_type.shape.dim.add()
        dim.dim_value = size
    
    # try out something with the conv layrs
    for node in graph.node:
        if node.op_type == 'Conv':
            print(f"Fixing up Conv layr: {node.name}")
            weight_name = node.input[1]  # Weights are usully the second imput
            weight_tensor = next((w for w in graph.initializer if w.name == weight_name), None)
            if weight_tensor:
                weights = numpy_helper.to_array(weight_tensor)
                print(f"  Old weights shape: {weights.shape}")
                weights = np.transpose(weights, (0, 2, 3, 1))  # Flip from MCHW to MHWC
                print(f"  New weights shape: {weights.shape}")
                graph.initializer.remove(weight_tensor)
                graph.initializer.append(numpy_helper.from_array(weights, weight_name))
            
            # chng the conv settins
            for attr in node.attribute:
                if attr.name == 'pads':
                    old_pads = list(attr.ints)
                    print(f"  Old padding: {old_pads}")
                    new_pads = [old_pads[0], old_pads[2], old_pads[1], old_pads[3]]  # Reordr for NHWC
                    print(f"  New padding: {new_pads}")
                    attr.ints[:] = new_pads
                elif attr.name == 'kernel_shape':
                    print(f"  Kernal size: {list(attr.ints)}")
                elif attr.name == 'strides':
                    print(f"  Strids: {list(attr.ints)}")
    
    # Right now this gives error
    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    return output_file

# Try to conver to NHWC, QONNX first, then manual if it flops
def make_it_nhwc(input_file, output_file):
    if qonnx_ready:
        try:
            model = ModelWrapper(input_file)
            converted = to_channels_last(model)
            converted.save(output_file)
            print("QONNX did the trik!")
            return output_file
        except Exception as uhoh:
            print(f"QONNX craped out: {uhoh}, switchin to manual mode")
    return flip_to_nhwc_the_hard_way(input_file, output_file)

# Main funcion to parse the model and spit out HLS4ML config
def get_hls_config(onnx_file):
    valid, nhwc, shape = check_model(onnx_file)
    if not valid:
        raise ValueError(f"Model’s messed up. Wantd 4D shape, got {shape}")
    
    model_file = onnx_file
    if not nhwc:
        print(f"Model’s in NCHW format {shape}. Let’s flip it to NHWC...")
        model_file = make_it_nhwc(onnx_file, onnx_file.replace('.onnx', '_nhwc.onnx'))
        print(f"New model savd at {model_file}")
    
    
    hls_settings = {
        'Model': {
            'Precision': 'ap_fixed<16,6>',  # 16-bit fixd point
            'ReuseFactor': 1 
        }
    }
    
    # Conver to HLS4ML
    try:
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model_file,
            output_dir='hls4ml_prj',
            part='xcvu9p-flgb2104-2-i',  
            hls_config=hls_settings,
            backend='Vivado'  # stick with Vivado
        )
        hls_model.compile() 
    except Exception as oof:
        raise RuntimeError(f"Couldn’t conver to HLS4ML: {oof}")
    
    
    config = {
        'model_name': hls_model.config.get_config_value(['Model', 'Name'], 'NoNameModel'),
        'input_shape': hls_model.graph.inputs[0].shape if hls_model.graph.inputs else 'Dunno',
        'output_shape': hls_model.graph.outputs[0].shape if hls_model.graph.outputs else 'Dunno',
        'precision': hls_model.config.get_config_value(['Model', 'Precision'], 'No clue'),
        'layers': []
    }
    
    
    for name, layer in hls_model.graph.items():
        layer_details = {
            'name': name,
            'type': layer.__class__.__name__,
            'input_shape': layer.inputs[0].shape if layer.inputs else 'Not shure',
            'output_shape': layer.shape
        }
        if hasattr(layer, 'weights'):
            if 'weight' in layer.weights:
                layer_details['weights_shape'] = layer.weights['weight'].shape
            if 'bias' in layer.weights:
                layer_details['bias_shape'] = layer.weights['bias'].shape
        config['layers'].append(layer_details)
    
    
    #config['performance'] = {
    #    'total_ops': hls_model.config.get_config_value(['EstimatedPerformance', 'Total_Operations'], 'No idea'),
    #    'total_latency': hls_model.config.get_config_value(['EstimatedPerformance', 'Total_Latency'], 'No idea')
    #}
    #config['resources'] = hls_model.config.get_config_value(['ResourceEstimation'], 'No clue')
    
    return config

def main(model_path):
    try:
        config = get_hls_config(model_path)
        import json
        print(json.dumps(config, indent=2))  # Prety print the result
        return config
    except Exception as dangit:
        print(f"Hit a snagg: {dangit}")
        raise

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Hey, give me an ONNX file to work with!")
