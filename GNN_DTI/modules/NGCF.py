import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np


class NGCF(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(NGCF, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.embedding_file = args_config.embedding_file

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self.embedding_dict, self.weight_dict = self.init_weight()

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def init_weight(self):

        # xavier init
        initializer = nn.init.xavier_uniform_

#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.n_users,
#                                                  self.emb_size))),
#             'item_emb': nn.Parameter(initializer(torch.empty(self.n_items,
#                                                  self.emb_size)))
#         })
        
        print(f"\nLoading LM embedding from: {self.embedding_file}")
        lm_embeddings = torch.load(self.embedding_file)

        with open('id_mappings.json', 'r') as f:
            id_mappings = json.load(f)

        embedding_dict = nn.ParameterDict()

        user_emb = torch.zeros((self.n_users, self.emb_size)).to(self.device)
        item_emb = torch.zeros((self.n_items, self.emb_size)).to(self.device)
        
        drug_embeddings = lm_embeddings['drug_embeddings'].to(self.device)
        for orig_id, mapped_id in id_mappings['drug'].items():
            orig_idx = lm_embeddings['drug_ids'].index(orig_id)
            user_emb[int(mapped_id)] = drug_embeddings[orig_idx]
        
        target_embeddings = lm_embeddings['target_embeddings'].to(self.device)
        for orig_id, mapped_id in id_mappings['target'].items():
            orig_idx = lm_embeddings['target_ids'].index(orig_id)
            item_emb[int(mapped_id)] = target_embeddings[orig_idx]
        
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        embedding_dict['user_emb'] = nn.Parameter(user_emb)
        embedding_dict['item_emb'] = nn.Parameter(item_emb)
        
        print(f"Completed embedding mapping:")
        print(f"Drug embedding shape: {user_emb.shape}")
        print(f"Target embedding shape: {item_emb.shape}")

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] * (self.context_hops+1)
        
        for k in range(self.context_hops):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def save_embeddings(self, path='trained_embeddings', with_id_mapping=True):
        print(f"\nSaving trained node embeddings")
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        user_embeddings = self.embedding_dict['user_emb'].detach().cpu().numpy()
        item_embeddings = self.embedding_dict['item_emb'].detach().cpu().numpy()
        
        np.save(os.path.join(path, 'user_embeddings.npy'), user_embeddings)
        np.save(os.path.join(path, 'item_embeddings.npy'), item_embeddings)
        
        user_gcn_emb, item_gcn_emb = self.generate(split=True)
        user_gcn_emb = user_gcn_emb.detach().cpu().numpy()
        item_gcn_emb = item_gcn_emb.detach().cpu().numpy()
        
        np.save(os.path.join(path, 'user_gcn_embeddings.npy'), user_gcn_emb)
        np.save(os.path.join(path, 'item_gcn_embeddings.npy'), item_gcn_emb)
        
        if with_id_mapping:
            try:
                with open('id_mappings.json', 'r') as f:
                    id_mappings = json.load(f)
                
                with open(os.path.join(path, 'id_mappings.json'), 'w') as f:
                    json.dump(id_mappings, f, indent=4)
            except Exception as e:
                print(f"Error saving ID mappings: {e}")
        
        print(f"Embeddings saved to {path} directory")
        print(f"User embedding shape: {user_embeddings.shape}")
        print(f"Item embedding shape: {item_embeddings.shape}")
        print(f"GCN user embedding shape: {user_gcn_emb.shape}")
        print(f"GCN item embedding shape: {item_gcn_emb.shape}")
        
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def gcn(self, edge_dropout=True, mess_dropout=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.edge_dropout_rate,
                                    self.sparse_norm_adj._nnz()) if edge_dropout else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(self.context_hops):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                            + self.weight_dict['b_bi_%d' % k]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            if mess_dropout:
                ego_embeddings = nn.Dropout(self.mess_dropout_rate)(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        return all_embeddings[:self.n_users, :], all_embeddings[self.n_users:, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(edge_dropout=False, mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)
        n_e = item_gcn_emb[neg_candidates]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e

        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])
        return neg_items_emb_[[[i] for i in range(batch_size)],
                              range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:
            return embeddings[:, -1, :]

    def forward(self, batch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_gcn_emb, item_gcn_emb = self.gcn(edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        pos_gcn_embs = item_gcn_emb[pos_item]

        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.negative_sampling(user_gcn_emb, item_gcn_emb,
                                                           user, neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                           pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user_gcn_emb[user], pos_gcn_embs, neg_gcn_embs)
