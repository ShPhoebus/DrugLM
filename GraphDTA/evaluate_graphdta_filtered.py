#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script: Evaluate a trained GraphDTA model on the filtered test set

Usage:
    python evaluate_graphdta_filtered.py \
        --test_file filter_graphdta_testset/GraphDTA_test_merged_filtered.csv \
        --model_file DrugLM/GraphDTA/model_val_lm_gin_GINConvNet.model \
        --embedding_file DrugLM/LM_finetune/e5_FT_embedding.pt \
        --output_file results_graphdta_filtered.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    average_precision_score, precision_recall_curve, auc
)

# Add GraphDTA directory to path
sys.path.insert(0, '/root/autodl-tmp/DrugLM/GraphDTA')
from models.ginconv import GINConvNet


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


def load_embeddings(embedding_file, lm_embedding_dim=1024):
    """Load LM embeddings"""
    drug_emb_map = {}
    prot_emb_map = {}

    if os.path.exists(embedding_file):
        print(f"Loading embeddings from {embedding_file}")
        embedding_dict = torch.load(embedding_file, map_location='cpu', weights_only=False)

        drug_tensor = embedding_dict.get('drug_embeddings', None)
        drug_ids = embedding_dict.get('drug_ids', [])

        prot_tensor = embedding_dict.get('target_embeddings', None)
        prot_ids = embedding_dict.get('target_ids', [])

        if drug_tensor is not None and drug_ids:
            for idx, drug_id in enumerate(drug_ids):
                drug_emb_map[str(drug_id)] = drug_tensor[idx].float()

        if prot_tensor is not None and prot_ids:
            for idx, prot_id in enumerate(prot_ids):
                prot_emb_map[str(prot_id)] = prot_tensor[idx].float()

        print(f"Loaded {len(drug_emb_map)} drug and {len(prot_emb_map)} protein embeddings")
    else:
        print(f"Warning: Embedding file not found: {embedding_file}")

    return drug_emb_map, prot_emb_map, lm_embedding_dim


def create_test_data(test_file, drug_emb_map, prot_emb_map, lm_embedding_dim):
    """Create test data"""
    print(f"Processing test data from {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"Total samples in test file: {len(test_df)}")

    # Build SMILES graph
    smile_graph = {}
    for sm in test_df["compound_iso_smiles"].unique():
        graph_data = smile_to_graph(sm)
        if graph_data is not None:
            smile_graph[sm] = graph_data

    # Filter invalid SMILES
    mask = test_df["compound_iso_smiles"].isin(smile_graph)
    test_df_filtered = test_df[mask].reset_index(drop=True)
    print(f"Valid samples with valid SMILES: {len(test_df_filtered)}/{len(test_df)}")

    zero_drug_emb = torch.zeros(lm_embedding_dim, dtype=torch.float)
    zero_prot_emb = torch.zeros(lm_embedding_dim, dtype=torch.float)

    test_data_list = []
    missing_drug = 0
    missing_prot = 0

    for i in range(len(test_df_filtered)):
        row = test_df_filtered.iloc[i]
        smiles = row["compound_iso_smiles"]
        sequence = row["target_sequence"]
        label = row["affinity"]
        drug_id = str(row["Compound_ID"])
        prot_id = str(row["Protein_ID"])

        features, edge_index = smile_graph[smiles]
        encoded_sequence = seq_cat(sequence)

        drug_lm = drug_emb_map.get(drug_id, zero_drug_emb)
        prot_lm = prot_emb_map.get(prot_id, zero_prot_emb)

        if torch.equal(drug_lm, zero_drug_emb):
            missing_drug += 1
        if torch.equal(prot_lm, zero_prot_emb):
            missing_prot += 1

        data = Data(
            x=features,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.float),
            target=encoded_sequence.unsqueeze(0),
            drug_lm_embedding=drug_lm,
            protein_lm_embedding=prot_lm
        )
        test_data_list.append(data)

    if missing_drug > 0 or missing_prot > 0:
        print(f"Missing embeddings: {missing_drug} drug, {missing_prot} protein")

    return test_data_list


def predicting(model, device, loader):
    """Predict"""
    model.eval()
    total_labels = []
    total_preds = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            total_preds.append(outputs.cpu().numpy())
            total_labels.append(data.y.cpu().numpy())

    total_labels = np.concatenate(total_labels)
    total_preds = np.concatenate(total_preds)

    return total_labels, total_preds.flatten()


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphDTA on filtered test set')
    parser.add_argument('--test_file', type=str, required=True, help='Path to filtered test CSV file')
    parser.add_argument('--model_file', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to embedding file')
    parser.add_argument('--output_file', type=str, default='results_filtered.csv', help='Output results file')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load embeddings
    drug_emb_map, prot_emb_map, lm_embedding_dim = load_embeddings(args.embedding_file)

    # Create test data
    test_data = create_test_data(args.test_file, drug_emb_map, prot_emb_map, lm_embedding_dim)
    print(f"Test data size: {len(test_data)}")

    # Create data loader
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Get feature dimensions
    num_features_xd = test_data[0].x.shape[1]
    print(f"Drug feature dimension: {num_features_xd}")
    print(f"LM embedding dimension: {lm_embedding_dim}")

    # Load model
    print(f"Loading model from {args.model_file}")
    model = GINConvNet(num_features_xd=num_features_xd, lm_embedding_dim=lm_embedding_dim, use_lm=True)
    model.load_state_dict(torch.load(args.model_file, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Predict
    print("Running predictions...")
    labels, probs = predicting(model, device, test_loader)

    # Compute metrics
    preds_binary = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds_binary)
    auc_score = roc_auc_score(labels, probs)
    aupr = average_precision_score(labels, probs)
    f1 = f1_score(labels, preds_binary)

    # AUPR computed from PR curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    aupr_curve = auc(recall, precision)

    # Statistics
    n_total = len(labels)
    n_positive = sum(labels == 1)
    n_negative = sum(labels == 0)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Filtered Test Set)")
    print("=" * 60)
    print(f"Test file: {args.test_file}")
    print(f"Model: {args.model_file}")
    print(f"Embedding: {args.embedding_file}")
    print("-" * 60)
    print(f"Total samples: {n_total}")
    print(f"Positive samples: {n_positive} ({n_positive/n_total*100:.1f}%)")
    print(f"Negative samples: {n_negative} ({n_negative/n_total*100:.1f}%)")
    print("-" * 60)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC:       {auc_score:.4f} ({auc_score*100:.2f}%)")
    print(f"AUPR (AP): {aupr:.4f} ({aupr*100:.2f}%)")
    print(f"AUPR (PR): {aupr_curve:.4f} ({aupr_curve*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("=" * 60)

    # Save results
    results_df = pd.DataFrame({
        'Label': labels,
        'Predicted_Probability': probs
    })
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to: {args.output_file}")

    # Save metrics summary
    summary_file = args.output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("EVALUATION RESULTS (Filtered Test Set)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test file: {args.test_file}\n")
        f.write(f"Model: {args.model_file}\n")
        f.write(f"Embedding: {args.embedding_file}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total samples: {n_total}\n")
        f.write(f"Positive samples: {n_positive} ({n_positive/n_total*100:.1f}%)\n")
        f.write(f"Negative samples: {n_negative} ({n_negative/n_total*100:.1f}%)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"AUC:       {auc_score:.4f} ({auc_score*100:.2f}%)\n")
        f.write(f"AUPR (AP): {aupr:.4f} ({aupr*100:.2f}%)\n")
        f.write(f"AUPR (PR): {aupr_curve:.4f} ({aupr_curve*100:.2f}%)\n")
        f.write(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)\n")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
