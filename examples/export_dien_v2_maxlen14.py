import numpy as np
import torch

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DIEN_ME
import torch.onnx
import torch.utils.data as Data


def get_amazon_data():
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

    return uids, mids, cats, mid_his, cat_his,lengths_x

def get_xy_fd(use_neg=False, hash_flag=False):
    feature_columns = [SparseFeat('user', 543060, embedding_dim=18, use_hash=hash_flag),
                       SparseFeat('item_id', 367983, embedding_dim=18, use_hash=hash_flag),
                       SparseFeat('cate_id', 1601, embedding_dim=18, use_hash=hash_flag),
                       DenseFeat('pay_score', 1)]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=367983, embedding_dim=18, embedding_name='item_id'),
                         maxlen=14, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=1601, embedding_dim=18, embedding_name='cate_id'),
                         maxlen=14,
                         length_name="seq_length")]

    behavior_feature_list = ["item_id", "cate_id"] # 表示这两个特征是用户行为，将作为key_feat
    uid,  item_id, cate_id, hist_item_id, hist_cate_id, behavior_length = get_amazon_data()  # hist_cate_id序列长347
    score = np.random.rand(16,1)


    feature_dict = {'user': uid,  'item_id': item_id, 'cate_id': cate_id,
                    'hist_item_id': hist_item_id, 'hist_cate_id': hist_cate_id,
                    'pay_score': score, "seq_length": behavior_length}

    # if use_neg:
    #     feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
    #     feature_dict['neg_hist_cate_id'] = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0], [1, 2, 0, 0]])
    #     feature_columns += [
    #         VarLenSparseFeat(
    #             SparseFeat('neg_hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
    #             maxlen=4, length_name="seq_length"),
    #         VarLenSparseFeat(
    #             SparseFeat('neg_hist_cate_id', vocabulary_size=2 + 1, embedding_dim=4, embedding_name='cate_id'),
    #             maxlen=4, length_name="seq_length")]

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}

    return x,  feature_columns, behavior_feature_list





if __name__ == "__main__":
    x, feature_columns, behavior_feature_list = get_xy_fd(use_neg=False)

    device = 'cpu'
    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

 

    model = DIEN_ME(feature_columns, behavior_feature_list,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.6, gru_type="AUGRU", use_negsampling=False, device=device)

    model.eval()
    # prepare for data input
    
    if isinstance(x, dict):
            x = [x[feature] for feature in model.feature_index]
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    # tensor_data = Data.TensorDataset(
    #         torch.from_numpy(np.concatenate(x, axis=-1)))
    # test_loader = DataLoader(
    #         dataset=tensor_data, shuffle=False, batch_size=batch_size)
    tensor_data = torch.from_numpy(np.concatenate(x, axis=-1))
    tensor_data = tensor_data[0:2]

    input_names = ['input']
    dynamic_axes = {
            'input': {
                0: 'batch'
 
            },
    }
    torch.onnx.export(
        model,
        tensor_data,
        'dien_dynamicbatch_maxlen14.onnx',
        input_names=input_names,
        output_names=['score'],
        verbose=True,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=True,
        dynamic_axes=dynamic_axes,
        opset_version=11
    )
