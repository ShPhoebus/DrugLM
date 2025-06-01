import numpy as np
import pandas as pd
import random
import torch
from collections import Counter
import json
import numpy as py
import sys
pt_file = sys.argv[1] if len(sys.argv) > 1 else 'bge_FT.pt'

def extract_sequence(fasta_str):
    lines = fasta_str.splitlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence


with open('../data/real_data_raw/id_mappings.json', "r") as file:
    data = json.load(file)
    drug_mapping_dict = data.get("drug", {})  # Extracts the drug dictionary
    target_mapping_dict = data.get("target", {})  # Extracts the target dictionary

# load the embedding data
embedding_data = torch.load(f'../data/interaction/our_data/real_data_raw/{pt_file}')
drug_embeddings = embedding_data['drug_embeddings']
target_embeddings = embedding_data['target_embeddings']
drug_ids = embedding_data['drug_ids']
target_ids = embedding_data['target_ids']

#load the datas
data_type = 'test'
data_dir = '../data/real_data_raw/'
drug_with_id = pd.read_csv(data_dir + 'drug_with_ID.csv', encoding='ISO-8859-1', low_memory=False)
target_with_id = pd.read_csv(data_dir + 'target_with_ID.csv', encoding='ISO-8859-1', low_memory=False)
drug_target_file = np.loadtxt(data_dir + data_type + '.txt', dtype=int).T
drug_target_file = drug_target_file.tolist()
output_dir = f'../data/interaction/our_data/{data_type}.txt'


# processing the protein
protein_list_raw = drug_target_file[1]
drug_list_raw = drug_target_file[0]

if data_type == 'train':
    protein_list = protein_list_raw
    drug_list = drug_list_raw
elif data_type == 'test':
    protein_list = []
    drug_list = []
    for i in range(len(protein_list_raw)):
        if str(protein_list_raw[i]) in target_ids and str(drug_list_raw[i]) in drug_ids:
            protein_list.append(protein_list_raw[i])
            drug_list.append(drug_list_raw[i])
else:
    raise ValueError("WTF")

print('# of proteins in {} dataset: '.format(data_type), len(protein_list))
protein_list = np.unique(protein_list)
'''
for i in protein_list:
    print(i)
'''
print('number of unique proteins: ', len(protein_list))

# get the protein sequences
filtered_sequences = target_with_id.set_index('GNN_ID').loc[protein_list, 'amino-acid-sequence'].tolist()
contains_empty_or_none_sequence = any(item == "" or pd.isna(item) for item in filtered_sequences)
print('does protein sequence contains illegal values?: ', contains_empty_or_none_sequence)
print('# of sequences acquired: ', len(filtered_sequences))
print('number of unique sequences: ', len(np.unique(filtered_sequences)))

# remove the duplicate sequences
protein_df = pd.DataFrame({"id": protein_list, "sequence": filtered_sequences})
# Drop duplicate sequences, keeping the first occurrence
protein_df = protein_df.drop_duplicates(subset="sequence", keep="first")
print('duplication removed!')
protein_df['raw_sequence'] = protein_df['sequence'].apply(extract_sequence)
protein_df.drop(columns=['sequence'], inplace=True)
protein_df.to_csv(f"../data/interaction/our_data/proteins_{data_type}.csv", index=False)

protein_dict = dict(zip(protein_df['id'], protein_df['raw_sequence']))
print()
print()

# processing the drugs

print('# of drugs in {} dataset: '.format(data_type), len(drug_list))
drug_list = np.unique(drug_list)
'''
for i in drug_list:
    print(i)
'''
print('number of unique drugs: ', len(drug_list))

# get the drug SMILES
filtered_smiles = drug_with_id.set_index('GNN_ID').loc[drug_list, 'SMILES'].tolist()
contains_empty_or_none_smiles = any(item == "" or pd.isna(item) for item in filtered_smiles)
print('does drug SMILES contains illegal values?: ', contains_empty_or_none_smiles)
print('# of SMILES acquired: ', len(filtered_smiles))
print('# of unique SMILES acquired: ', len(np.unique(filtered_smiles)))

# save the drug smiles
drug_df = pd.DataFrame({'id': drug_list, 'SMILES': filtered_smiles})
drug_df = drug_df.dropna(subset=['SMILES'])
print('duplication and illegal values removed!')
drug_df.to_csv(f"../data/interaction/our_data/drugs_{data_type}.csv", index=False)

drug_dict = dict(zip(drug_df['id'], drug_df['SMILES']))
print('# of SMILES acquired: ', len(filtered_smiles))
print()
print()

drug_ids = drug_target_file[0]
protein_ids = drug_target_file[1]
# generate dataset
positive_samples = [
    (drug_dict[drug_id], protein_dict[protein_id], 1)
    for drug_id, protein_id in zip(drug_ids, protein_ids)
    if drug_id in drug_dict and protein_id in protein_dict
]

# Create a set of known interactions to avoid duplicate negatives
known_interactions = set(zip(drug_ids, protein_ids))

# Generate negative samples equal in size to positive samples
all_drug_ids = list(drug_dict.keys())
all_protein_ids = list(protein_dict.keys())

negative_samples = set()

while len(negative_samples) < len(positive_samples):
    neg_drug = random.choice(all_drug_ids)
    neg_protein = random.choice(all_protein_ids)

    # Ensure it's not a known interaction and exists in both dictionaries
    if (neg_drug, neg_protein) not in known_interactions:
        negative_samples.add((neg_drug, neg_protein))

# Format negative samples with label 0
negative_samples = [
    (drug_dict[drug_id], protein_dict[protein_id], 0)
    for drug_id, protein_id in negative_samples
    if drug_id in drug_dict and protein_id in protein_dict
]

# Step 3: Combine positive and negative samples
dataset = positive_samples + negative_samples
random.shuffle(dataset)  # Shuffle to mix positive and negative samples

# Step 4: Write to train.txt
with open(output_dir, 'w') as f:
    for smile, sequence, label in dataset:
        f.write(f"{smile},{sequence},{label}\n")

print(f"{data_type}.txt has been created successfully!")


