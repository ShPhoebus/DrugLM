import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
import argparse
from torch_geometric.data import Data
from utils import TestbedDataset

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                            ['C','N','O','S','F','Cl','Br','I','H','P', 'Unknown']) +
                    one_of_k_encoding_unk(atom.GetDegree(), list(range(0, 11))) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(0, 11))) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(0, 11))) +
                    [atom.GetIsAromatic()], dtype=np.float32)

def smile_to_graph(smile: str):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except:
        pass

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None
    features = [atom_features(a) for a in mol.GetAtoms()]
    
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    
    if len(edges) == 0 and num_atoms > 1:
        return None

    features_tensor = torch.tensor(np.array(features), dtype=torch.float)
    if edges:
        edge_index_tensor = torch.tensor(np.array(edges).T, dtype=torch.long).contiguous()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

    return features_tensor, edge_index_tensor

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {c: i+1 for i, c in enumerate(seq_voc)}
max_seq_len = 1000

def seq_cat(seq: str):
    if pd.isnull(seq):
        seq = ""
    encoded = np.zeros(max_seq_len, dtype=int)
    for i, ch in enumerate(seq[:max_seq_len]):
        encoded[i] = seq_dict.get(ch, 0)
    return torch.tensor(encoded, dtype=torch.long)

def parse_args():
    parser = argparse.ArgumentParser(description='Create PyTorch Geometric data with LM embeddings.')
    parser.add_argument('--raw_dir', type=str, default='data/mydata', help='Directory containing the merged CSV files')
    parser.add_argument('--proc_dir', type=str, default='data/processed', help='Directory to save processed .pt files')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to the .pt file containing LM embeddings')
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Dimension of the LM embeddings')
    parser.add_argument('--train_file', type=str, default='train_merged.csv', help='Name of the training CSV file')
    parser.add_argument('--test_file', type=str, default='test_merged.csv', help='Name of the test CSV file')
    parser.add_argument('--output_train_file', type=str, default='mydata_train_lm.pt', help='Output filename for processed training data')
    parser.add_argument('--output_test_file', type=str, default='mydata_test_lm.pt', help='Output filename for processed test data')
    return parser.parse_args()

args = parse_args()

RAW_DIR = args.raw_dir
PROC_DIR = args.proc_dir
EMBEDDING_FILE = args.embedding_file
LM_EMBEDDING_DIM = args.embedding_dim

os.makedirs(PROC_DIR, exist_ok=True)

print(f"Loading embeddings from {EMBEDDING_FILE}")
drug_emb_map = {}
prot_emb_map = {}

if os.path.exists(EMBEDDING_FILE):
    try:
        embedding_dict = torch.load(EMBEDDING_FILE, map_location='cpu')

        drug_tensor = embedding_dict.get('drug_embeddings', None)
        drug_ids = embedding_dict.get('drug_ids', [])
        if drug_tensor is None:
            for key, value in embedding_dict.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 2 and key != 'target_embeddings':
                    drug_tensor = value
                    break

        prot_tensor = embedding_dict.get('target_embeddings', None)
        prot_ids = embedding_dict.get('target_ids', [])
        if prot_tensor is None:
             for key, value in embedding_dict.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 2 and key != 'drug_embeddings':
                    prot_tensor = value
                    break

        if drug_tensor is not None and drug_tensor.shape[1] != LM_EMBEDDING_DIM:
            LM_EMBEDDING_DIM = drug_tensor.shape[1]
        if prot_tensor is not None and prot_tensor.shape[1] != LM_EMBEDDING_DIM:
            LM_EMBEDDING_DIM = prot_tensor.shape[1]

        if drug_tensor is not None and drug_ids:
            for idx, drug_id in enumerate(drug_ids):
                drug_emb_map[str(drug_id)] = drug_tensor[idx].float()

        if prot_tensor is not None and prot_ids:
            for idx, prot_id in enumerate(prot_ids):
                prot_emb_map[str(prot_id)] = prot_tensor[idx].float()

        print(f"Loaded {len(drug_emb_map)} drug and {len(prot_emb_map)} protein embeddings")

    except Exception as e:
        print(f"Error loading embedding file: {e}")
else:
    print(f"Warning: Embedding file not found, using zero vectors")

zero_drug_emb = torch.zeros(LM_EMBEDDING_DIM, dtype=torch.float)
zero_prot_emb = torch.zeros(LM_EMBEDDING_DIM, dtype=torch.float)

print(f"Processing training data...")
train_df = pd.read_csv(os.path.join(RAW_DIR, args.train_file))

smile_graph_train = {}
for sm in train_df["compound_iso_smiles"].unique():
    graph_data = smile_to_graph(sm)
    if graph_data is not None:
        smile_graph_train[sm] = graph_data

mask_train = train_df["compound_iso_smiles"].isin(smile_graph_train)
train_df_filtered = train_df[mask_train].reset_index(drop=True)
print(f"Training: {len(train_df_filtered)}/{len(train_df)} samples with valid SMILES")

train_data_list = []
missing_drug_emb_train = 0
missing_prot_emb_train = 0

for i in range(len(train_df_filtered)):
    row = train_df_filtered.iloc[i]
    smiles = row["compound_iso_smiles"]
    sequence = row["target_sequence"]
    label = row["affinity"]
    drug_id = str(row["Compound_ID"])
    prot_id = str(row["Protein_ID"])

    features, edge_index = smile_graph_train[smiles]

    encoded_sequence = seq_cat(sequence)

    drug_lm = drug_emb_map.get(drug_id, zero_drug_emb)
    prot_lm = prot_emb_map.get(prot_id, zero_prot_emb)
    if torch.equal(drug_lm, zero_drug_emb):
        missing_drug_emb_train += 1
    if torch.equal(prot_lm, zero_prot_emb):
        missing_prot_emb_train += 1

    data = Data(
        x=features,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.float),
        target=encoded_sequence.unsqueeze(0),
        drug_lm_embedding=drug_lm,
        protein_lm_embedding=prot_lm
    )
    train_data_list.append(data)

if missing_drug_emb_train > 0 or missing_prot_emb_train > 0:
    print(f"Training: {missing_drug_emb_train} missing drug, {missing_prot_emb_train} missing protein embeddings")

print(f"Processing test data...")
test_df = pd.read_csv(os.path.join(RAW_DIR, args.test_file))

smile_graph_test = {}
for sm in test_df["compound_iso_smiles"].unique():
    graph_data = smile_to_graph(sm)
    if graph_data is not None:
        smile_graph_test[sm] = graph_data

mask_test = test_df["compound_iso_smiles"].isin(smile_graph_test)
test_df_filtered = test_df[mask_test].reset_index(drop=True)
print(f"Test: {len(test_df_filtered)}/{len(test_df)} samples with valid SMILES")

test_data_list = []
missing_drug_emb_test = 0
missing_prot_emb_test = 0

for i in range(len(test_df_filtered)):
    row = test_df_filtered.iloc[i]
    smiles = row["compound_iso_smiles"]
    sequence = row["target_sequence"]
    label = row["affinity"]
    drug_id = str(row["Compound_ID"])
    prot_id = str(row["Protein_ID"])

    features, edge_index = smile_graph_test[smiles]

    encoded_sequence = seq_cat(sequence)

    drug_lm = drug_emb_map.get(drug_id, zero_drug_emb)
    prot_lm = prot_emb_map.get(prot_id, zero_prot_emb)
    if torch.equal(drug_lm, zero_drug_emb):
        missing_drug_emb_test += 1
    if torch.equal(prot_lm, zero_prot_emb):
        missing_prot_emb_test += 1

    data = Data(
        x=features,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.float),
        target=encoded_sequence.unsqueeze(0),
        drug_lm_embedding=drug_lm,
        protein_lm_embedding=prot_lm
    )
    test_data_list.append(data)

if missing_drug_emb_test > 0 or missing_prot_emb_test > 0:
    print(f"Test: {missing_drug_emb_test} missing drug, {missing_prot_emb_test} missing protein embeddings")

output_train_path = os.path.join(PROC_DIR, args.output_train_file)
output_test_path = os.path.join(PROC_DIR, args.output_test_file)

torch.save(train_data_list, output_train_path)
torch.save(test_data_list, output_test_path)

print(f"Saved {len(train_data_list)} training and {len(test_data_list)} test samples")