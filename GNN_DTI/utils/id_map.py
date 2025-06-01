import numpy as np
import json

def create_id_mappings(train_file):
    train_data = np.loadtxt(train_file, dtype=np.int32)

    all_drug_ids = set(int(x) for x in train_data[:, 0])
    all_target_ids = set(int(x) for x in train_data[:, 1])

    drug_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_drug_ids))}
    target_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_target_ids))}

    mappings = {
        'drug': drug_id_map,
        'target': target_id_map
    }
    
    with open('id_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    return drug_id_map, target_id_map

def load_id_mappings():
    with open('id_mappings.json', 'r') as f:
        mappings = json.load(f)

    drug_map = {int(k): v for k, v in mappings['drug'].items()}
    target_map = {int(k): v for k, v in mappings['target'].items()}
    
    return drug_map, target_map

if __name__ == "__main__":
    train_file = "data/drug/train.txt"
    drug_map, target_map = create_id_mappings(train_file)
