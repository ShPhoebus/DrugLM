import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

n_users = 0
n_items = 0

def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    return feed_dict


if __name__ == '__main__':
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    save_emb_epochs = []
    if args.save_emb_epochs:
        try:
            save_emb_epochs = [int(e) for e in args.save_emb_epochs.split(',')]
            print(f"Will save embeddings at epochs: {save_emb_epochs}")
        except:
            print("Error parsing save_emb_epochs parameter, format should be comma-separated numbers, e.g. '10,20,30'")

    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    from modules.LightGCN import LightGCN
    from modules.NGCF import NGCF
    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    else:
        model = NGCF(n_params, args, norm_mat).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    best_epoch = 0
    best_test_ret = None
    best_test_recall20 = 0
    optimal_threshold = 0.0

    print("start training ...")
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        if epoch % 5 == 0:
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio", "AUC", "AUPR", "Bal AUC", "Bal AUPR", "ACC Full", "ACC Bal"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test', threshold=optimal_threshold)
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio'], test_ret['auc'], test_ret['aupr'],
                 test_ret.get('balanced_auc', 0.), test_ret.get('balanced_aupr', 0.),
                 test_ret.get('acc_full', 0.), test_ret.get('acc_balanced', 0.)])

            recall_20_idx = 3
            if test_ret['recall'][recall_20_idx] > best_test_recall20:
                best_test_recall20 = test_ret['recall'][recall_20_idx]
                best_epoch = epoch
                best_test_ret = test_ret

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret 
            else:
                test_s_t = time()
                valid_ret, optimal_threshold = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio'], valid_ret['auc'], valid_ret['aupr'],
                     valid_ret.get('balanced_auc', 0.), valid_ret.get('balanced_aupr', 0.),
                     valid_ret.get('acc_full', 0.), valid_ret.get('acc_balanced', 0.)])
            print(train_res)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][recall_20_idx], 
                                                                      cur_best_pre_0,
                                                                      stopping_step, 
                                                                      expected_order='acc',
                                                                      flag_step=10)

            if epoch in save_emb_epochs:
                print(f'\nSaving node embeddings for epoch {epoch}...')
                model.eval()
                embeddings_path = args.out_dir + f'epoch_{epoch}_embeddings'
                if hasattr(model, 'save_embeddings') and callable(model.save_embeddings):
                    model.save_embeddings(path=embeddings_path)
                    print(f'Epoch {epoch} embeddings saved to: {embeddings_path}')
                else:
                    print(f"Model {type(model).__name__} does not have save_embeddings method, skipping save.")
                if not should_stop:
                    model.train()

            if should_stop:
                break

            if valid_ret['recall'][recall_20_idx] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
        else:
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d' % (epoch))
    print('\nBest performance:')
    print(f'Best epoch: {best_epoch}')
    
    print('\nSaving best model node embeddings...')
    if args.save:
        if os.path.exists(args.out_dir + 'model_' + '.ckpt'):
            model.load_state_dict(torch.load(args.out_dir + 'model_' + '.ckpt'))
            print(f'Loaded best model weights: {args.out_dir}model_.ckpt')
    
    model.eval()
    embeddings_path = args.out_dir + 'best_model_embeddings'
    if hasattr(model, 'save_embeddings') and callable(model.save_embeddings):
        model.save_embeddings(path=embeddings_path)
        print(f'Best model embeddings saved to: {embeddings_path}')
    else:
        print(f"Model {type(model).__name__} does not have save_embeddings method, skipping best embedding save.")
    
    print('\nMetrics@5:')
    print(f'Recall@5: {best_test_ret["recall"][1]:.4f}')
    print(f'NDCG@5: {best_test_ret["ndcg"][1]:.4f}')
    print(f'Precision@5: {best_test_ret["precision"][1]:.4f}')
    print(f'HR@5: {best_test_ret["hit_ratio"][1]:.4f}')
    
    print('\nMetrics@10:')
    print(f'Recall@10: {best_test_ret["recall"][2]:.4f}')
    print(f'NDCG@10: {best_test_ret["ndcg"][2]:.4f}')
    print(f'Precision@10: {best_test_ret["precision"][2]:.4f}')
    print(f'HR@10: {best_test_ret["hit_ratio"][2]:.4f}')
    
    print('\nMetrics@20:')
    print(f'Recall@20: {best_test_ret["recall"][3]:.4f}')
    print(f'NDCG@20: {best_test_ret["ndcg"][3]:.4f}')
    print(f'Precision@20: {best_test_ret["precision"][3]:.4f}')
    print(f'HR@20: {best_test_ret["hit_ratio"][3]:.4f}')
    
    if best_test_ret and 'auc' in best_test_ret and 'aupr' in best_test_ret:
        print('\nGlobal Metrics:')
        print(f'AUC: {best_test_ret["auc"]:.4f}')
        print(f'AUPR: {best_test_ret["aupr"]:.4f}')
        
        if 'balanced_auc' in best_test_ret and 'balanced_aupr' in best_test_ret:
            print(f'Balanced AUC: {best_test_ret["balanced_auc"]:.4f}')
            print(f'Balanced AUPR: {best_test_ret["balanced_aupr"]:.4f}')
        if 'acc_full' in best_test_ret and 'acc_balanced' in best_test_ret:
            print(f'ACC Full: {best_test_ret["acc_full"]:.4f}')
            print(f'ACC Balanced: {best_test_ret["acc_balanced"]:.4f}')
