import onnx
import onnxruntime
import numpy as np
import onnx_graphsurgeon as gs
import onnx.shape_inference as shape_inference

model_path = "./dien_dynamicbatch.onnx"
onnxruntime.set_default_logger_severity(3)
session = onnxruntime.InferenceSession(model_path)
print("session created")

infered_onnx_model = shape_inference.infer_shapes(onnx.load(model_path))
onnx.save_model(infered_onnx_model, model_path.split('/')[-1].split('.')[0] + '_shape.onnx')

gs_graph = gs.import_onnx(onnx.load(model_path))
org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
inputs = dict()
for inp in session.get_inputs():
    # inputs[inp.name] = np.random.random(size=inp.shape).astype(org_inputs[inp.name]['dtype'])
    a = np.load('data_batch2.npy')
    if len(a.shape)<2:
        a = np.expand_dims(a, 0)
    inputs[inp.name] = a.repeat(2,axis=0)
outputs = [x.name for x in session.get_outputs()]

print('inputs:',{k:v.shape for k,v in inputs.items()})
print('outputs names:', outputs)
ort_outputs = session.run(outputs, inputs)
print(ort_outputs)
