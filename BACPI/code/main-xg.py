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
from data_process import training_data_process


args = argparse.ArgumentParser(description='Argparse for compound-protein interactions prediction')
args.add_argument('-task', type=str, default='interaction', help='affinity/interaction')
args.add_argument('-dataset', type=str, default='human', help='choose a dataset')
args.add_argument('-mode', type=str, default='gpu', help='gpu/cpu')
args.add_argument('-cuda', type=str, default='0', help='visible cuda devices')
args.add_argument('-verbose', type=int, default=1, help='0: do not output log in stdout, 1: output log')

args.add_argument('-lr', type=float, default=0.0005, help='init learning rate')
args.add_argument('-step_size', type=int, default=10, help='step size of lr_scheduler')
args.add_argument('-gamma', type=float, default=0.5, help='lr weight decay rate')
args.add_argument('-batch_size', type=int, default=4, help='batch size')
args.add_argument('-num_epochs', type=int, default=20, help='number of epochs')

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

args.add_argument("-our_data", action="store_true", help='weather or not to use embedding')
args.add_argument('-llm_model', type=str, default='bge_embeddings_step_78320epoch10_noSMILES2SeqInChI.pt', help='what llm model to aid ')

params, _ = args.parse_known_args()


def train_eval(model, task, data_train, data_dev, data_test, device, params):
    if task == 'affinity':
        criterion = F.mse_loss
        best_res = 2 ** 10
    elif task == 'interaction':
        criterion = F.cross_entropy
        best_res = 0
    else:
        print("Please choose a correct mode!!!")
        return 
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    idx = np.arange(len(data_train[0]))
    batch_size = params.batch_size
    if params.our_data:
        embedding_data = torch.load(f'../data/interaction/our_data/{params.llm_model}')
        drug_embeddings = embedding_data['drug_embeddings']
        target_embeddings = embedding_data['target_embeddings']
        drug_ids = embedding_data['drug_ids']
        target_ids = embedding_data['target_ids']

        drug_embedding_list = drug_embeddings.tolist()
        drug_dict = dict(zip(drug_ids, drug_embedding_list))

        target_embedding_list = target_embeddings.tolist()
        target_dict = dict(zip(target_ids, target_embedding_list))

    else:
        drug_dict = None
        target_dict = None

    for epoch in range(params.num_epochs):
        print('epoch: {}'.format(epoch))
        np.random.shuffle(idx)
        model.train()
        pred_labels = []
        predictions = []
        labels = []
        for i in range(math.ceil(len(data_train[0]) / batch_size)):
            batch_data = [data_train[di][idx[i * batch_size: (i + 1) * batch_size]] for di in range(len(data_train))]
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label, raw_protein_ids, raw_drug_ids \
                = batch2tensor(batch_data, device, params.our_data)
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, params, device,
                         raw_protein_ids, raw_drug_ids, target_dict, drug_dict)
            if task == 'affinity':
                loss = criterion(pred.float(), label.float())
                predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
                labels += label.cpu().numpy().reshape(-1).tolist()
            elif task == 'interaction':
                loss = criterion(pred.float(), label.view(label.shape[0]).long())
                ys = F.softmax(pred, 1).to('cpu').data.numpy()
                pred_labels += list(map(lambda x: np.argmax(x), ys))
                predictions += list(map(lambda x: x[1], ys))
                labels += label.cpu().numpy().reshape(-1).tolist()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if params.verbose:
                sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'.format(epoch, i, math.ceil(len(data_train[0])/batch_size)-1, float(loss.data)))
                sys.stdout.flush()

        if task == 'affinity':
            print(' ')
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse_train, pearson_train, spearman_train = regression_scores(labels, predictions)
            print('Train rmse:{}, pearson:{}, spearman:{}'.format(rmse_train, pearson_train, spearman_train))

            rmse_dev, pearson_dev, spearman_dev = test(model, task, data_dev, batch_size, device, params)
            rmse_test, pearson_test, spearman_test = test(model, task, data_test, batch_size, device, params)
            print('Dev rmse:{}, pearson:{}, spearman:{}'.format(rmse_dev, pearson_dev, spearman_dev))
            print( 'Test rmse:{}, pearson:{}, spearman:{}'.format(rmse_test, pearson_test, spearman_test))

            if rmse_dev < best_res:
                best_res = rmse_dev
                # torch.save(model, '../checkpoint/best_model_affinity.pth')
                res = [rmse_test, pearson_test, spearman_test]
        
        else:
            print(' ')
            pred_labels = np.array(pred_labels)
            predictions = np.array(predictions)
            labels = np.array(labels)
            auc_train, acc_train, apur_train = classification_scores(labels, predictions, pred_labels)
            print('Train auc:{}, acc:{}, aupr:{}'.format(auc_train, acc_train, apur_train))


            if params.our_data:
                auc_dev, acc_dev, aupr_dev = test(model, task, data_dev, batch_size, device, params, target_dict, drug_dict)
                auc_test, acc_test, aupr_test = test(model, task, data_test, batch_size, device, params, target_dict, drug_dict)
            else:
                auc_dev, acc_dev, aupr_dev = test(model, task, data_dev, batch_size, device, params)
                auc_test, acc_test, aupr_test = test(model, task, data_test, batch_size, device, params)

            print('Dev auc:{}, acc:{}, aupr:{}'.format(auc_dev, acc_dev, aupr_dev))
            print('Test auc:{}, acc:{}, aupr:{}'.format(auc_test, acc_test, aupr_test))

            if auc_dev > best_res:
                best_res = auc_dev
                # torch.save(model, '../checkpoint/best_model_interaction.pth')
                res = [auc_test, acc_test, aupr_test]

        scheduler.step()
    return res


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
