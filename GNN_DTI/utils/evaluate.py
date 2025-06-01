from .metrics import *
from .parser import parse_args

import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    rating = x[0]
    u = x[1]
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, user_dict, n_params, mode='test', threshold=None):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.,
              'aupr': 0.,
              'balanced_auc': 0.,
              'balanced_aupr': 0.,
              'acc_full': 0.,
              'acc_balanced': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']

    all_pred_scores_list = []
    all_true_labels_list = []

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    user_gcn_emb, item_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start:end]
        if not user_list_batch:
            continue
            
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        current_batch_ratings_for_all_items_np = None
        if batch_test_flag:
            current_batch_ratings_for_all_items_np = np.zeros(shape=(len(user_batch), n_items))
            i_count = 0
            if i_batch_size > 0:
                n_item_batchs = n_items // i_batch_size + 1
            else:
                n_item_batchs = 0 if n_items == 0 else 1
            
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)
                if i_start >= i_end:
                    continue

                item_batch_ids = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(device)
                i_g_embeddings_batch = item_gcn_emb[item_batch_ids]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embeddings_batch).detach().cpu().numpy()
                current_batch_ratings_for_all_items_np[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]
            if n_items > 0:
                assert i_count == n_items, f"Item count mismatch: {i_count} vs {n_items}"
        else:
            current_batch_ratings_for_all_items_np = model.rating(u_g_embeddings, item_gcn_emb).detach().cpu().numpy()

        if current_batch_ratings_for_all_items_np is not None:
            for u_idx_in_batch, user_id in enumerate(user_list_batch):
                user_scores_for_all_items = current_batch_ratings_for_all_items_np[u_idx_in_batch]
                user_positive_items_in_test = test_user_set.get(user_id, [])
                user_items_in_train = train_user_set.get(user_id, [])

                for item_id in range(n_items):
                    if item_id not in user_items_in_train:
                        score = user_scores_for_all_items[item_id]
                        label = 1 if item_id in user_positive_items_in_test else 0
                        all_pred_scores_list.append(score)
                        all_true_labels_list.append(label)

        user_batch_rating_uid = zip(current_batch_ratings_for_all_items_np, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']
            result['recall'] += re['recall']
            result['ndcg'] += re['ndcg']
            result['hit_ratio'] += re['hit_ratio']

    if n_test_users > 0:
        result['precision'] /= n_test_users
        result['recall'] /= n_test_users
        result['ndcg'] /= n_test_users
        result['hit_ratio'] /= n_test_users

    if len(all_true_labels_list) > 0:
        final_true_labels = np.array(all_true_labels_list)
        final_pred_scores = np.array(all_pred_scores_list)

        result['auc'] = AUC(final_true_labels, final_pred_scores)
        result['aupr'] = AUPR(final_true_labels, final_pred_scores)

        pos_indices = np.where(final_true_labels == 1)[0]
        neg_indices = np.where(final_true_labels == 0)[0]
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)
        
        can_create_balanced_set = num_pos > 0 and num_neg >= num_pos
        balanced_true_labels_for_opt = None
        balanced_pred_scores_for_opt = None

        if can_create_balanced_set:
            np.random.seed(42)
            sampled_neg_indices = np.random.choice(neg_indices, size=num_pos, replace=False)
            balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
            balanced_pred_scores = final_pred_scores[balanced_indices]
            balanced_true_labels = final_true_labels[balanced_indices]
            if mode == 'valid':
                balanced_true_labels_for_opt = balanced_true_labels
                balanced_pred_scores_for_opt = balanced_pred_scores
 
        if mode == 'valid':
            best_acc_balanced_valid = -1
            best_threshold_for_acc = 0.0

            if can_create_balanced_set and balanced_pred_scores_for_opt is not None:
                min_score_bal, max_score_bal = np.min(balanced_pred_scores_for_opt), np.max(balanced_pred_scores_for_opt)
                if np.isfinite(min_score_bal) and np.isfinite(max_score_bal) and max_score_bal > min_score_bal:
                    thresholds_to_try = np.linspace(min_score_bal, max_score_bal, 21)
                    for th in thresholds_to_try:
                        pred_labels_bal_opt = (balanced_pred_scores_for_opt > th).astype(int)
                        current_acc_bal = ACC(balanced_true_labels_for_opt, pred_labels_bal_opt)
                        if current_acc_bal > best_acc_balanced_valid:
                            best_acc_balanced_valid = current_acc_bal
                            best_threshold_for_acc = th
                print(f"[VALIDATION] Best threshold for ACC_balanced: {best_threshold_for_acc:.4f} with ACC_balanced: {best_acc_balanced_valid:.4f}")
            else:
                print("[VALIDATION] Cannot create balanced set for threshold optimization. Using default threshold.")

        current_threshold = best_threshold_for_acc if mode == 'valid' else threshold
        if current_threshold is None: 
             print("[WARNING] Threshold is None in test mode. Defaulting to 0.0 for ACC calculation.")
             current_threshold = 0.0
             
        pred_labels_full = (final_pred_scores > current_threshold).astype(int)
        result['acc_full'] = ACC(final_true_labels, pred_labels_full)

        if can_create_balanced_set:
            result['balanced_auc'] = AUC(balanced_true_labels, balanced_pred_scores)
            result['balanced_aupr'] = AUPR(balanced_true_labels, balanced_pred_scores)
            pred_labels_balanced = (balanced_pred_scores > current_threshold).astype(int)
            result['acc_balanced'] = ACC(balanced_true_labels, pred_labels_balanced)
        else:
            result['balanced_auc'] = 0.
            result['balanced_aupr'] = 0.
            result['acc_balanced'] = 0.
            
    else:
        result['auc'] = 0.
        result['aupr'] = 0.
        result['balanced_auc'] = 0.
        result['balanced_aupr'] = 0.
        result['acc_full'] = 0.
        result['acc_balanced'] = 0.

    if n_test_users > 0:
        assert count == n_test_users, f"Count mismatch: {count} vs {n_test_users}"
    pool.close()

    if mode == 'valid':
        return result, best_threshold_for_acc
    else:
        return result
