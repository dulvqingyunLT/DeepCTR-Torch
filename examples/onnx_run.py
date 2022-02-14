import onnx
import onnxruntime
import numpy as np
import onnx_graphsurgeon as gs
import onnx.shape_inference as shape_inference


def get_amazon_data_prepared():
    uids = np.load('./one_batch_data/uids.npy')
    mids = np.load('./one_batch_data/mids.npy')
    cats = np.load('./one_batch_data/cats.npy')
    # mid_his = np.load('./one_batch_data/mid_his.npy')
    # cat_his = np.load('./one_batch_data/cat_his.npy')
    mid_his = np.random.randint(0,10000,(16,14), dtype=np.int32)
    cat_his = np.random.randint(0,1000,(16,14), dtype=np.int32)
    mid_mask = np.load('./one_batch_data/mid_mask.npy')
    target = np.load('./one_batch_data/target.npy')
    # lengths_x = np.load('./one_batch_data/lengths_x.npy')
    lengths_x = np.array([14,12,11,10,8,8,7,6,6,6,5,5,5,5,5,5])
    noclk_mid_his = np.load('./one_batch_data/noclk_mid_his.npy')
    noclk_cat_his = np.load('./one_batch_data/noclk_cat_his.npy')

    score = np.random.rand(16,1)

    x = [uids,mids,cats,score,mid_his,lengths_x,cat_his]

    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    return np.concatenate(x, axis=-1)


model_path = "./dien_dynamicbatch_maxlen14.onnx"
onnxruntime.set_default_logger_severity(3)
session = onnxruntime.InferenceSession(model_path)
print("session created")

infered_onnx_model = shape_inference.infer_shapes(onnx.load(model_path))
onnx.save_model(infered_onnx_model, model_path.split('/')[-1].split('.')[0] + '_shape.onnx')

gs_graph = gs.import_onnx(onnx.load(model_path))
org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
inputs = dict()
for inp in session.get_inputs():
    # a = np.load('./first_edition/data_batch2.npy')
    # if len(a.shape)<2:
    #     a = np.expand_dims(a, 0)
    # inputs[inp.name] = a.repeat(2,axis=0)

    a = get_amazon_data_prepared()
    inputs[inp.name] = a[0:2]
outputs = [x.name for x in session.get_outputs()]

print('inputs:',{k:v.shape for k,v in inputs.items()})
print('outputs names:', outputs)
ort_outputs = session.run(outputs, inputs)
print(ort_outputs)
