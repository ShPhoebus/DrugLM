import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

class GraphConv(nn.Module):
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):

        ego_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = ego_embed
        embs = [ego_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1) 
        return embs[:self.n_users, :], embs[self.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN, self).__init__()

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

        self._init_weight(data_config)
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()

    def _init_weight(self, data_config):
    
#       Random embedding
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        
        print(f"\nLoading LM embedding from: {self.embedding_file}")
        lm_embeddings = torch.load(self.embedding_file)

        
        with open('id_mappings.json', 'r') as f:
            id_mappings = json.load(f)

        self.user_embed = lm_embeddings['drug_embeddings']
        self.item_embed = lm_embeddings['target_embeddings']

        drug_embeddings = lm_embeddings['drug_embeddings'].to(self.device)
        for orig_id, mapped_id in id_mappings['drug'].items():
            orig_idx = lm_embeddings['drug_ids'].index(orig_id)
            self.user_embed[int(mapped_id)] = drug_embeddings[orig_idx]

        target_embeddings = lm_embeddings['target_embeddings'].to(self.device)
        for orig_id, mapped_id in id_mappings['target'].items():
            orig_idx = lm_embeddings['target_ids'].index(orig_id)
            self.item_embed[int(mapped_id)] = target_embeddings[orig_idx]

        self.user_embed = F.normalize(self.user_embed, p=2, dim=1)
        self.item_embed = F.normalize(self.item_embed, p=2, dim=1)

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        
        
    def save_embeddings(self, path='trained_embeddings', with_id_mapping=True):

        if not os.path.exists(path):
            os.makedirs(path)

        user_embeddings = self.user_embed.detach().cpu().numpy()
        item_embeddings = self.item_embed.detach().cpu().numpy()

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
                print(f"save error: {e}")
        
        print(f"Saved embedding in {path}")


    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        if self.ns == 'rns':
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
        else:
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.negative_sampling(user_gcn_emb, item_gcn_emb,
                                                           user, neg_item[:, k*self.n_negs: (k+1)*self.n_negs],
                                                           pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
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

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

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
