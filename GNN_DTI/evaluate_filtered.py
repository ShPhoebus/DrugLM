#!/usr/bin/env python
"""
Evaluation script: Load a trained model and evaluate on the filtered test set
Usage: python evaluate_filtered.py --model-dir ./weights_e5_pre --gnn lightgcn --embedding-file /path/to/embedding.pt
"""
import os
import sys

# ====== Set CUDA environment variables first ======
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ====== Change to working directory ======
os.chdir('/root/autodl-tmp/DrugLM/GNN_DTI')
sys.path.insert(0, os.getcwd())

# ====== Then import other modules ======
import json
import numpy as np
import torch
from collections import defaultdict
import scipy.sparse as sp

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

from utils.metrics import AUC, AUPR, ACC
from utils.parser import parse_args


def load_id_mappings():
    """Load ID mappings"""
    with open('id_mappings.json', 'r') as f:
        id_mappings = json.load(f)
        drug_map = {int(k): int(v) for k, v in id_mappings['drug'].items()}
        target_map = {int(k): int(v) for k, v in id_mappings['target'].items()}
    return drug_map, target_map


def read_cf_mapped(file_name, drug_map, target_map):
    """Read and map data"""
    raw_data = np.loadtxt(file_name, dtype=np.int32)
    mapped_data = []
    for u, i in raw_data:
        if u in drug_map and i in target_map:
            mapped_data.append([drug_map[u], target_map[i]])
    return np.array(mapped_data)


def load_filtered_test_data(filtered_test_path, drug_map, target_map):
    """Load filtered test set"""
    test_user_set = defaultdict(list)
    test_cf = read_cf_mapped(filtered_test_path, drug_map, target_map)
    for u, i in test_cf:
        test_user_set[u].append(i)
    return test_user_set, test_cf


def build_sparse_graph_from_train(train_path, drug_map, target_map, n_users, n_items):
    """Build sparse graph from training data"""
    train_cf = read_cf_mapped(train_path, drug_map, target_map)

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
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]
    cf_ = np.concatenate([cf, cf_], axis=0)

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users + n_items, n_users + n_items))

    return _bi_norm_lap(mat), train_cf


def load_train_data(train_path, drug_map, target_map):
    """Load training data"""
    train_user_set = defaultdict(list)
    train_cf = read_cf_mapped(train_path, drug_map, target_map)
    for u, i in train_cf:
        train_user_set[u].append(i)
    return train_user_set, train_cf


def evaluate_on_test(model, test_user_set, train_user_set, n_users, n_items, device, threshold=0.0):
    """Evaluate model on test set"""
    model.eval()

    all_pred_scores = []
    all_true_labels = []

    user_gcn_emb, item_gcn_emb = model.generate()

    test_users = list(test_user_set.keys())
    batch_size = 1024

    with torch.no_grad():
        for start in range(0, len(test_users), batch_size):
            end = min(start + batch_size, len(test_users))
            user_list_batch = test_users[start:end]

            user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
            u_g_embeddings = user_gcn_emb[user_batch]

            # Get scores for all items
            ratings = model.rating(u_g_embeddings, item_gcn_emb).detach().cpu().numpy()

            for u_idx, user_id in enumerate(user_list_batch):
                user_scores = ratings[u_idx]
                user_positive_items = set(test_user_set[user_id])
                user_train_items = set(train_user_set.get(user_id, []))

                for item_id in range(n_items):
                    if item_id not in user_train_items:
                        score = user_scores[item_id]
                        label = 1 if item_id in user_positive_items else 0
                        all_pred_scores.append(score)
                        all_true_labels.append(label)

    all_pred_scores = np.array(all_pred_scores)
    all_true_labels = np.array(all_true_labels)

    # Compute global metrics
    auc = AUC(all_true_labels, all_pred_scores)
    aupr = AUPR(all_true_labels, all_pred_scores)

    # Compute balanced metrics
    pos_indices = np.where(all_true_labels == 1)[0]
    neg_indices = np.where(all_true_labels == 0)[0]

    if len(pos_indices) > 0 and len(neg_indices) >= len(pos_indices):
        np.random.seed(42)
        sampled_neg = np.random.choice(neg_indices, size=len(pos_indices), replace=False)
        balanced_indices = np.concatenate([pos_indices, sampled_neg])
        balanced_labels = all_true_labels[balanced_indices]
        balanced_scores = all_pred_scores[balanced_indices]

        balanced_auc = AUC(balanced_labels, balanced_scores)
        balanced_aupr = AUPR(balanced_labels, balanced_scores)
        pred_labels_balanced = (balanced_scores > threshold).astype(int)
        acc_balanced = ACC(balanced_labels, pred_labels_balanced)
    else:
        balanced_auc = 0.
        balanced_aupr = 0.
        acc_balanced = 0.

    # Compute full ACC
    pred_labels_full = (all_pred_scores > threshold).astype(int)
    acc_full = ACC(all_true_labels, pred_labels_full)

    return {
        'auc': auc,
        'aupr': aupr,
        'balanced_auc': balanced_auc,
        'balanced_aupr': balanced_aupr,
        'acc_full': acc_full,
        'acc_balanced': acc_balanced,
        'n_test_samples': len(all_true_labels),
        'n_positive': len(pos_indices)
    }


def main():
    # Manually parse extra parameters
    model_dir = None
    filtered_test = '/root/autodl-tmp/filter_gnndti_testset/GNN_DTI_test.txt'
    threshold = 0.0

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model-dir':
            model_dir = sys.argv[i + 1]
            sys.argv.pop(i)
            sys.argv.pop(i)
        elif sys.argv[i] == '--filtered-test':
            filtered_test = sys.argv[i + 1]
            sys.argv.pop(i)
            sys.argv.pop(i)
        elif sys.argv[i] == '--threshold':
            threshold = float(sys.argv[i + 1])
            sys.argv.pop(i)
            sys.argv.pop(i)
        else:
            i += 1

    if model_dir is None:
        print("Error: --model-dir parameter is required")
        sys.exit(1)

    # Use full parse_args
    args = parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    # Load ID mappings
    drug_map, target_map = load_id_mappings()
    n_users = len(drug_map)
    n_items = len(target_map)
    print(f"n_users: {n_users}, n_items: {n_items}")

    # Training data path
    train_path = args.data_path + args.dataset + '/train.txt'

    # Load training data (for building graph and excluding training samples)
    train_user_set, train_cf = load_train_data(train_path, drug_map, target_map)
    print(f"Training samples: {len(train_cf)}")

    # Build normalized graph
    norm_mat, _ = build_sparse_graph_from_train(train_path, drug_map, target_map, n_users, n_items)

    # Load filtered test set
    filtered_test_user_set, filtered_test_cf = load_filtered_test_data(filtered_test, drug_map, target_map)
    print(f"Filtered test samples: {len(filtered_test_cf)}")

    # Prepare model parameters
    n_params = {
        'n_users': n_users,
        'n_items': n_items,
        'pretrain_drug_emb': None,
        'pretrain_target_emb': None,
    }

    # Load embeddings (if provided)
    if args.embedding_file and os.path.exists(args.embedding_file):
        try:
            data = torch.load(args.embedding_file, map_location='cpu', weights_only=False)
            # Check key name format
            if 'drug_embeddings' in data:
                drug_embeddings = data['drug_embeddings']
                target_embeddings = data['target_embeddings']
                drug_ids = data['drug_ids']
                target_ids = data['target_ids']
            elif 'drug' in data:
                drug_embeddings = data['drug']['embeddings']
                target_embeddings = data['target']['embeddings']
                drug_ids = data['drug']['ids']
                target_ids = data['target']['ids']
            else:
                raise KeyError(f"Unknown embedding file format, keys: {data.keys()}")

            # Map to correct IDs
            new_drug_emb = torch.zeros((len(drug_map), drug_embeddings.shape[1]))
            new_target_emb = torch.zeros((len(target_map), target_embeddings.shape[1]))

            for old_id, new_id in drug_map.items():
                if old_id in drug_ids:
                    old_idx = list(drug_ids).index(old_id)
                    new_drug_emb[new_id] = drug_embeddings[old_idx]

            for old_id, new_id in target_map.items():
                if old_id in target_ids:
                    old_idx = list(target_ids).index(old_id)
                    new_target_emb[new_id] = target_embeddings[old_idx]

            n_params['pretrain_drug_emb'] = new_drug_emb
            n_params['pretrain_target_emb'] = new_target_emb
            args.no_lm = False
            print(f"Loaded embeddings: {args.embedding_file}")
            print(f"Drug embedding shape: {new_drug_emb.shape}")
            print(f"Target embedding shape: {new_target_emb.shape}")
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            args.no_lm = True

    # Create model
    if args.gnn == 'lightgcn':
        from modules.LightGCN import LightGCN
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        from modules.NGCF import NGCF
        model = NGCF(n_params, args, norm_mat).to(device)

    # Load model weights
    model_path = os.path.join(model_dir, 'model_.ckpt')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model: {model_path}")

    # Evaluate
    results = evaluate_on_test(model, filtered_test_user_set, train_user_set,
                               n_users, n_items, device, threshold)

    print("\n" + "="*50)
    print(f"Model directory: {model_dir}")
    print(f"Filtered test set: {filtered_test}")
    print("="*50)
    print(f"AUC: {results['auc']:.4f}")
    print(f"AUPR: {results['aupr']:.4f}")
    print(f"Balanced AUC: {results['balanced_auc']:.4f}")
    print(f"Balanced AUPR: {results['balanced_aupr']:.4f}")
    print(f"ACC Full: {results['acc_full']:.4f}")
    print(f"ACC Balanced: {results['acc_balanced']:.4f}")
    print(f"Test samples: {results['n_test_samples']}")
    print(f"Positive samples: {results['n_positive']}")
    print("="*50)


if __name__ == '__main__':
    main()
