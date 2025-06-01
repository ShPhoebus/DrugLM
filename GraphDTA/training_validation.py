import numpy as np
import pandas as pd
import sys, os
import argparse
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train and validate GraphDTA model with LM embeddings for binary classification.')
    parser.add_argument('model_idx', type=int, help='Index for model selection (0:GIN, 1:GAT, 2:GAT_GCN, 3:GCN)')
    parser.add_argument('gpu_idx', type=int, help='GPU index to use (e.g., 0)')
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Dimension of the LM embeddings')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=512, help='Validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=512, help='Testing batch size')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing processed data (.pt files)')
    parser.add_argument('--train_dataset_file', type=str, default='mydata_train_lm.pt', help='Filename of the processed training data (used for train/val split)')
    parser.add_argument('--test_dataset_file', type=str, default='mydata_test_lm.pt', help='Filename of the processed test data')
    parser.add_argument('--save_model_name', type=str, default='model_val_lm', help='Base name for saving the best model based on validation')
    parser.add_argument('--result_file_name', type=str, default='result_val_lm', help='Base name for saving the final test results')
    parser.add_argument('--val_split_ratio', type=float, default=0.2, help='Ratio of training data to use for validation')
    parser.add_argument('--log_interval', type=int, default=20, help='Log training progress every N batches')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--no_lm', action='store_true', help='Do NOT concatenate LM embeddings in the model.')
    return parser.parse_args()

args = parse_args()
use_lm = not args.no_lm

modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.model_idx]
model_st = modeling.__name__

cuda_name = f"cuda:{args.gpu_idx}"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print(f'Device: {device}, Model: {model_st}, LM: {use_lm}')
if use_lm:
    print(f'LM embedding dim: {args.embedding_dim}')

TRAIN_BATCH_SIZE = args.train_batch_size
VAL_BATCH_SIZE = args.val_batch_size
TEST_BATCH_SIZE = args.test_batch_size
LR = args.lr
LOG_INTERVAL = args.log_interval
NUM_EPOCHS = args.epochs
DATA_DIR = args.data_dir
TRAIN_FILE = args.train_dataset_file
TEST_FILE = args.test_dataset_file
LM_EMBEDDING_DIM = args.embedding_dim
SAVE_MODEL_NAME = args.save_model_name
RESULT_FILE_NAME = args.result_file_name
VAL_SPLIT_RATIO = args.val_split_ratio
NUM_WORKERS = args.num_workers

train_val_data_path = os.path.join(DATA_DIR, TRAIN_FILE)
test_data_path = os.path.join(DATA_DIR, TEST_FILE)

if not os.path.isfile(train_val_data_path):
    print(f'Error: Training data not found at {train_val_data_path}')
    print('Please run create_data.py first with the --embedding_file argument.')
    sys.exit(1)
if not os.path.isfile(test_data_path):
    print(f'Error: Test data not found at {test_data_path}')
    print('Please run create_data.py first with the --embedding_file argument.')
    sys.exit(1)

print(f"Loading data...")
train_val_data = torch.load(train_val_data_path)
test_data = torch.load(test_data_path)

num_train_val = len(train_val_data)
num_val = int(num_train_val * VAL_SPLIT_RATIO)
num_train = num_train_val - num_val

train_data, valid_data = torch.utils.data.random_split(train_val_data, [num_train, num_val])

print(f'Train: {len(train_data)}, Val: {len(valid_data)}, Test: {len(test_data)}')

train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

if train_data:
    num_features_xd = train_data.dataset[train_data.indices[0]].x.shape[1]
else:
    num_features_xd = 45

model = modeling(num_features_xd=num_features_xd, lm_embedding_dim=LM_EMBEDDING_DIM, use_lm=use_lm).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=False)

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    total_loss = 0
    processed_samples = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.view(-1, 1).float()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        processed_samples += data.num_graphs

        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch {epoch} [{processed_samples}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)] Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}: Avg Loss: {avg_loss:.6f}')

def predicting_and_evaluating(model, device, loader):
    model.eval()
    total_logits = []
    total_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_logits.append(output.cpu())
            total_labels.append(data.y.view(-1, 1).cpu())

    logits_tensor = torch.cat(total_logits, dim=0)
    labels_tensor = torch.cat(total_labels, dim=0)

    metrics = {}
    try:
        probs = torch.sigmoid(logits_tensor).numpy().flatten()
        labels = labels_tensor.numpy().flatten()
        preds = (probs > 0.5).astype(int)

        metrics['accuracy'] = accuracy_score(labels, preds)
        if len(np.unique(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, probs)
            metrics['f1'] = f1_score(labels, preds)
        else:
            metrics['auc'] = 0.5
            metrics['f1'] = f1_score(labels, preds, zero_division=0)

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics['accuracy'] = 0.0
        metrics['auc'] = 0.0
        metrics['f1'] = 0.0

    return metrics, labels, probs

best_val_auc = 0.0
best_epoch = -1
lm_suffix = "" if use_lm else "_noLM"
model_file_name = f"{SAVE_MODEL_NAME}_{model_st}{lm_suffix}.pt"
result_file_name = f"{RESULT_FILE_NAME}_{model_st}{lm_suffix}.csv"

print(f"Training for {NUM_EPOCHS} epochs...")

for epoch in range(1, NUM_EPOCHS + 1):
    train(model, device, train_loader, optimizer, epoch, LOG_INTERVAL)

    val_metrics, _, _ = predicting_and_evaluating(model, device, valid_loader)
    val_auc = val_metrics.get('auc', 0.0)
    print(f"Val - Acc: {val_metrics.get('accuracy', 0.0):.4f}, AUC: {val_auc:.4f}, F1: {val_metrics.get('f1', 0.0):.4f}")

    scheduler.step(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        torch.save(model.state_dict(), model_file_name)
        print(f'New best AUC: {best_val_auc:.4f} at epoch {epoch}')

print(f"\nBest validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

print(f"Loading best model for test evaluation...")
model = modeling(num_features_xd=num_features_xd, lm_embedding_dim=LM_EMBEDDING_DIM, use_lm=use_lm).to(device)
model.load_state_dict(torch.load(model_file_name))

test_metrics, test_labels, test_probs = predicting_and_evaluating(model, device, test_loader)

print(f"\nTest Results:")
print(f"  Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
print(f"  AUC:      {test_metrics.get('auc', 0.0):.4f}")
print(f"  F1 Score: {test_metrics.get('f1', 0.0):.4f}")

results_df = pd.DataFrame({
    'Label': test_labels,
    'Predicted_Probability': test_probs
})
results_df.to_csv(result_file_name, index=False)
print(f"Results saved to {result_file_name}")

