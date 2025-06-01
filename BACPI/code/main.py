import os
import sys
import math
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import BACPI
from utils import *
from utils import create_batches
from data_process import training_data_process


args = argparse.ArgumentParser(description='Argparse for compound-protein interactions prediction')
args.add_argument('-task', type=str, default='interaction', help='affinity/interaction')
args.add_argument('-dataset', type=str, default='human', help='choose a dataset')
args.add_argument('-mode', type=str, default='gpu', help='gpu/cpu')
args.add_argument('-cuda', type=str, default='0', help='visible cuda devices')
args.add_argument('-verbose', type=int, default=1, help='0: do not output log in stdout, 1: output log')

# Hyper-parameter
args.add_argument('-lr', type=float, default=0.0005, help='init learning rate')
args.add_argument('-step_size', type=int, default=10, help='step size of lr_scheduler')
args.add_argument('-gamma', type=float, default=0.5, help='lr weight decay rate')
args.add_argument('-batch_size', type=int, default=4, help='batch size')
args.add_argument('-num_epochs', type=int, default=20, help='number of epochs')

# graph attention layer
args.add_argument('-gat_dim', type=int, default=50, help='dimension of node feature in graph attention layer')
args.add_argument('-num_head', type=int, default=3, help='number of graph attention layer head')
args.add_argument('-dropout', type=float, default=0.1)
args.add_argument('-alpha', type=float, default=0.1, help='LeakyReLU alpha')

args.add_argument('-comp_dim', type=int, default=80, help='dimension of compound atoms feature')
args.add_argument('-prot_dim', type=int, default=80, help='dimension of protein amino feature')
args.add_argument('-latent_dim', type=int, default=80, help='dimension of compound and protein feature')

args.add_argument('-window', type=int, default=5, help='window size of cnn model')
args.add_argument('-layer_cnn', type=int, default=3, help='number of layer in cnn model')
args.add_argument('-layer_out', type=int, default=3, help='number of output layer in prediction model')

# our data layer
args.add_argument("-our_data", action="store_true", help='weather or not to use embedding')
args.add_argument('-llm_model', type=str, default='bge_embeddings_step_78320epoch10_noSMILES2SeqInChI.pt', help='what llm model to aid ')

params, _ = args.parse_known_args()

def train_eval(model, task, train_data, dev_data, test_data, device, params):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)

    if params.our_data:
        embedding_data = torch.load(f'../data/interaction/our_data/real_data_raw/{params.llm_model}')
        drug_dict = dict(zip(embedding_data['drug_ids'], embedding_data['drug_embeddings'].tolist()))
        target_dict = dict(zip(embedding_data['target_ids'], embedding_data['target_embeddings'].tolist()))
    else:
        drug_dict = target_dict = None

    for epoch in range(params.num_epochs):
        model.train()
        total_loss = 0
        train_batches = create_batches(train_data, params.batch_size)

        for batch in train_batches:
            atoms, atoms_mask, adjacencies, fps, amino, amino_mask, label, raw_protein_ids, raw_drug_ids = batch2tensor(batch, device, params.our_data)
            pred = model(atoms, atoms_mask, adjacencies, amino, amino_mask, fps, params, device,
                         raw_protein_ids, raw_drug_ids, target_dict, drug_dict)
            loss = F.cross_entropy(pred, label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_batches)
        # print(f"epoch {epoch}, avg loss: {avg_loss:.4f}")
        scheduler.step()
        avg_loss = total_loss / len(train_batches)
        print(f"epoch {epoch}, avg loss: {avg_loss:.4f}", flush=True)

    

    print()
    print("evaluating...")

    model.eval()
    pred_list = []
    label_list = []

    test_batches = create_batches(test_data, params.batch_size)
    for batch in test_batches:
        atoms, atoms_mask, adjacencies, fps, amino, amino_mask, label, raw_protein_ids, raw_drug_ids = batch2tensor(batch, device, params.our_data)
        with torch.no_grad():
            pred = model(atoms, atoms_mask, adjacencies, amino, amino_mask, fps, params, device,
                         raw_protein_ids, raw_drug_ids, target_dict, drug_dict)
        pred_prob = F.softmax(pred, dim=1).cpu().numpy()[:, 1]
        pred_label = np.argmax(pred.cpu().numpy(), axis=1)
        label = label.cpu().numpy()

        pred_list += list(pred_prob)
        label_list += list(label)

    auc, acc, aupr = classification_scores(np.array(label_list), np.array(pred_list), np.array(pred_list) > 0.5)
    print(f"Finally test result of auc: {auc}, acc: {acc}, aupr: {aupr}")

    return auc, acc, aupr


def test(model, task, data_test, batch_size, device, params, target_dict=None, drug_dict=None):
    model.eval()
    predictions = []
    pred_labels = []
    labels = []
    for i in range(math.ceil(len(data_test[0]) / batch_size)):
        batch_data = [data_test[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_test))]
        atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label, raw_protein_ids, raw_drug_ids \
            = batch2tensor(batch_data, device, params.our_data)
        with torch.no_grad():
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, params, device,
                         raw_protein_ids, raw_drug_ids, target_dict, drug_dict)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().numpy().reshape(-1).tolist()
        else:
            ys = F.softmax(pred, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))
            predictions += list(map(lambda x: x[1], ys))
            labels += label.cpu().numpy().reshape(-1).tolist()
    pred_labels = np.array(pred_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)
    if task == 'affinity':
        rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
        return rmse_value, pearson_value, spearman_value
    else:
        auc_value, acc_value, aupr_value = classification_scores(labels, predictions, pred_labels)
        return auc_value, acc_value, aupr_value


if __name__ == '__main__':
    
    print(params)
    task = params.task
    dataset = params.dataset
    if params.our_data:
        print("Using LLM embedding!")
        print(f'embedding: {params.llm_model}')
    else:
        print("running without embedding")

    data_dir = '../datasets/' + task + '/' + dataset
    if not os.path.isdir(data_dir):
        training_data_process(task, dataset)

    if params.mode == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("cuda is not available!!!")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print('The code run on the', device)

    print('Load data...')
    train_data = load_data(data_dir, 'train', params.our_data)
    test_data = load_data(data_dir, 'test', params.our_data)
    train_data, dev_data = split_data(train_data, 0.1)

    atom_dict = pickle.load(open(data_dir + '/atom_dict', 'rb'))
    amino_dict = pickle.load(open(data_dir + '/amino_dict', 'rb'))

    print('training...')
    model = BACPI(task, len(atom_dict), len(amino_dict), params)
    model.to(device)
    res = train_eval(model, task, train_data, dev_data, test_data, device, params)

    print('Finish training!')
    if task == 'affinity':
        print('Finally test result of rmse:{}, pearson:{}, spearman:{}'.format(res[0], res[1], res[2]))
    elif task == 'interaction':
        print('Finally test result of auc:{}, acc:{}, aupr:{}'.format(res[0], res[1], res[2]))
