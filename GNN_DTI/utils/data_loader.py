import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf_amazon(file_name, drug_map=None, target_map=None):
    raw_data = np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]
    
    if drug_map is not None and target_map is not None:
        mapped_data = []
        for u, i in raw_data:
            if u in drug_map and i in target_map:
                mapped_data.append([drug_map[u], target_map[i]])
        return np.array(mapped_data)
    return raw_data


def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def load_pretrained_embeddings(pretrain_path, drug_map=None, target_map=None):
    try:
        data = torch.load(pretrain_path)
        drug_embeddings = torch.FloatTensor(data['drug']['embeddings'])
        target_embeddings = torch.FloatTensor(data['target']['embeddings'])
        drug_ids = data['drug']['ids']
        target_ids = data['target']['ids']

        if drug_map is not None and target_map is not None:
            new_drug_embeddings = torch.zeros((len(drug_map), drug_embeddings.shape[1]))
            new_target_embeddings = torch.zeros((len(target_map), target_embeddings.shape[1]))

            for old_id, new_id in drug_map.items():
                if old_id in drug_ids:
                    old_idx = list(drug_ids).index(old_id)
                    new_drug_embeddings[new_id] = drug_embeddings[old_idx]
                    
            for old_id, new_id in target_map.items():
                if old_id in target_ids:
                    old_idx = list(target_ids).index(old_id)
                    new_target_embeddings[new_id] = target_embeddings[old_idx]
            
            return new_drug_embeddings, new_target_embeddings, drug_map, target_map
            
        return drug_embeddings, target_embeddings, {}, {}
    except Exception as e:
        print(f"Error loading pretrained embeddings: {e}")
        return None, None, {}, {}


def build_sparse_graph(train_cf):
    total_nodes = n_users + n_items
    
    def _bi_norm_lap(adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    cf = train_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))

    return _bi_norm_lap(mat)


def analyze_target_coverage(directory):
    train_data = np.loadtxt(directory + 'train.txt', dtype=np.int32)
    test_data = np.loadtxt(directory + 'test.txt', dtype=np.int32)

    train_targets = set(train_data[:, 1])
    test_targets = set(test_data[:, 1])

    missing_targets = test_targets - train_targets

    import json
    try:
        with open('id_mappings.json', 'r') as f:
            id_mappings = json.load(f)
            drug_map = {int(k): int(v) for k, v in id_mappings['drug'].items()}
            target_map = {int(k): int(v) for k, v in id_mappings['target'].items()}

        mapped_train = read_cf_amazon(directory + 'train.txt', drug_map, target_map)
        mapped_test = read_cf_amazon(directory + 'test.txt', drug_map, target_map)

        mapped_train_targets = set(mapped_train[:, 1])
        mapped_test_targets = set(mapped_test[:, 1])
        mapped_missing_targets = mapped_test_targets - mapped_train_targets
            
    except Exception as e:
        print(f"Error during mapped ID analysis: {e}")
    
    return missing_targets


def load_data(model_args):
    global args, dataset, n_users, n_items
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'
    
    import json
    try:
        with open('id_mappings.json', 'r') as f:
            id_mappings = json.load(f)
            drug_map = {int(k): int(v) for k, v in id_mappings['drug'].items()}
            target_map = {int(k): int(v) for k, v in id_mappings['target'].items()}
    except Exception as e:
        print(f"Error loading ID mappings: {e}")
        raise

    global n_users, n_items
    n_users = len(drug_map)  # 7188
    n_items = len(target_map)  # 4142
    
    train_cf = read_cf_amazon(directory + 'train.txt', drug_map, target_map)
    test_cf = read_cf_amazon(directory + 'test.txt', drug_map, target_map)
    valid_cf = read_cf_amazon(directory + 'valid.txt', drug_map, target_map)
 
    for u, i in train_cf:
        train_user_set[u].append(i)
    for u, i in test_cf:
        test_user_set[u].append(i)
    for u, i in valid_cf:
        valid_user_set[u].append(i)

    norm_mat = build_sparse_graph(train_cf)
    
    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'pretrain_drug_emb': None,  
        'pretrain_target_emb': None,
        'drug_id_to_idx': {},
        'target_id_to_idx': {}
    }
    
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set,
        'test_user_set': test_user_set,
    }

    return train_cf, user_dict, n_params, norm_mat

