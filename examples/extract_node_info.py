import onnx
import onnxruntime
from onnx import helper
import onnx_graphsurgeon as gs
from collections import OrderedDict
import numpy as np
import argparse
import json
import os

def extract_const_shape(nodes):
    """nodes: onnx_graphsurgeon nodes"""
    const_shapes = dict()
    for node in nodes:
        for inp in node.inputs:
            if type(inp).__name__ == 'Constant':
                const_shapes[inp.name] = list(inp.shape)
    return const_shapes

def extract_node_info(model_path, input_shapes=None, input_data=None, dynamic_input_shape=False):
    """
    model: onnx model path
    input_shapes: input shape dict
    input_data: input data dict
    dynamic_input_shape: True if dynamic model
    """
    print(f'Loading ONNX model from {model_path}')
    model = onnx.load(model_path)
    assert(isinstance(model, onnx.ModelProto))
    onnx.checker.check_model(model_path)

    gs_graph = gs.import_onnx(model)
    node_info = OrderedDict()

    print('Extracting constant shapes')
    const_shapes = extract_const_shape(gs_graph.nodes)

    org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
    print('org_inputs',org_inputs)
    org_outputs = [out.name for out in gs_graph.outputs]
    count = 0
    for node in model.graph.node:
        node_info[node.name] = dict()
        attrs = []
        for att in node.attribute:
            attrs.append(str(att).replace("\n"," "))
        node_info[node.name]['op_type'] = node.op_type
        node_info[node.name]['attribute'] = attrs
        node_info[node.name]['inputs'] = dict()
        for inp in node.input:
            if inp in const_shapes:
                node_info[node.name]['inputs'][inp] = const_shapes[inp]
            else:
                node_info[node.name]['inputs'][inp] = [-1]
        node_info[node.name]['outputs'] = dict()
        for output in node.output:
            node_info[node.name]['outputs'][output] = [-1]
            value_info = helper.ValueInfoProto()
            value_info.name = output
            model.graph.output.append(value_info)
            count+=1
    print(f'{count} extra nodes are marked')

    # export intermediate model
    inter_model_dir = os.path.join('./inter_model', model_path.split('/')[-1].split('.')[0])
    inter_model_path = os.path.join(inter_model_dir, model_path.split('/')[-1])
    if not os.path.exists(inter_model_dir):
        os.makedirs(inter_model_dir)
    try:
        onnx.save_model(model, inter_model_path)
    except:
        onnx.save_model(model, inter_model_path, save_as_external_data=True, location='external_weight', all_tensors_to_one_file=False)
    print(f'Intermediate model saved at {inter_model_path}')

    import gc
    del model
    gc.collect()
    print("after gc work")
    # prepare for onnx runtime
    onnxruntime.set_default_logger_severity(3)
    session = onnxruntime.InferenceSession(inter_model_path)
    if input_data:
        ort_inputs = input_data
    else:
        ort_inputs = dict()
        if not dynamic_input_shape:
            for inp in gs_graph.inputs:
                input_shapes[inp.name] = inp.shape
        for i, inp in enumerate(session.get_inputs()):
            for i in range(len(input_shapes[inp.name])): 
                if input_shapes[inp.name][i] == -1:
                    input_shapes[inp.name][i] = org_inputs[inp.name]['shape'][i]
                assert(isinstance(input_shapes[inp.name][i], int)), \
                        f'{input_shapes[inp.name]} got wrong dim {input_shapes[inp.name][i]}'
            shape = tuple(input_shapes[inp.name])
            ort_inputs[inp.name] = np.random.random(size=shape).astype(org_inputs[inp.name]['dtype'])
    outputs = [x.name for x in session.get_outputs()]

    # onnx runtime
    print('ort_inputs:',{k:v.shape for k,v in ort_inputs.items()})
    ort_outputs = session.run(outputs, ort_inputs)
    ort_outputs = OrderedDict(zip(outputs, ort_outputs))

    print('Updating outputs shapes...')
    for node_name,values in node_info.items():
        for out in values['outputs']:
            node_info[node_name]['outputs'][out] = ort_outputs[out].shape

    print('Updating inputs shapes...')
    for node_name,values in node_info.items():
        for inp in values['inputs']:
            if inp in input_shapes:
                node_info[node_name]['inputs'][inp] = input_shapes[inp]
            if inp in ort_outputs:
                node_info[node_name]['inputs'][inp] = ort_outputs[inp].shape
    return node_info

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--model_path', required=True,
                        help='onnx model path')
    parser.add_argument('--output_dir', required=True,
                        help='the output dir to save the node_info json files')
    parser.add_argument('--input_shape', type=str, nargs='+',
                        help='The manually-set static input shape, useful when the input shape is dynamic. '
                        'The value should be "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, '
                        'for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron '
                        'to make sure what the input name and dimension ordering (NCHW or NHWC) is.')
    parser.add_argument('--input_data_path', type=str, nargs='+',
                        help='input data, The value should be "input_name1:xxx1.bin"  "input_name2:xxx2.bin ...", input data should be a binary data file.')
    parser.add_argument('--dynamic_input_shape', action='store_true',
                        help='This option enables dynamic input shape support.')
    return parser

def main():
    parser = argparse.ArgumentParser(
        description='Extract node information from ONNX model')
    parser = parse_args(parser)
    args = parser.parse_args()

    if args.dynamic_input_shape:
        if args.input_shape is None and args.input_data_path is None:
            raise RuntimeError(
                'Please specify input shape or input data for dynamic model. Run "python3 onnx_node_info.py -h" for details.'
                )

    input_shapes = dict()
    if args.input_shape is not None:
        for x in args.input_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes.update({name: shape})

    input_data_paths = dict()
    if args.input_data_path is not None:
        for x in args.input_data_path:
            pieces = x.split(':')
            name, data = ':'.join(pieces[:-1]), pieces[-1]
            input_data_paths.update({name: data})

    input_tensors = dict()
    if len(input_data_paths) > 0 and args.input_shape is not None:
        for name in input_shapes.keys():
            # input_data = np.fromfile(input_data_paths[name], dtype=np.float32)
            input_data = np.load(input_data_paths[name])
            input_data = input_data.reshape(input_shapes[name])
            input_tensors.update({name: input_data})

    node_info = extract_node_info(
                    model_path = args.model_path,
                    input_shapes = input_shapes,
                    input_data = input_tensors,
                    dynamic_input_shape = args.dynamic_input_shape
            )

    # save node info to json
    json_file_name = args.model_path.split('/')[-1].split('.')[0] + '.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    json_path = os.path.join(args.output_dir,json_file_name)
    with open(json_path,'w') as f:
        json.dump(node_info,f)
        print(f'node_info saved at {json_path}')

if __name__=='__main__':
    # python ./utils/extract_node_info.py --model_path=./maskrcnn_r50_update2.onnx --output_dir=./node_info --input_shape input:1,3,1333,800 --dynamic_input_shape
    main()