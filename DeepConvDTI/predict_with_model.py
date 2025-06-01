import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import keras
from keras import backend as K
from keras.models import load_model
import argparse
import h5py
import os
import tensorflow as tf

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}


def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0] 
    else:
        return [seq_dic[aa] for aa in seq]

def parse_data(dti_dir, drug_dir, protein_dir, with_label=True,
               prot_len=2500, prot_vec="Convolution",
               drug_vec="morgan_fp_r2", drug_len=2048,
               lm_embedding_size=768, drug_lm_file=None, protein_lm_file=None, embedding_file=None):
    # Handle single embedding file
    if embedding_file and os.path.exists(embedding_file):
        print(f"Using embedding file: {embedding_file}")
        drug_lm_file = embedding_file
        protein_lm_file = embedding_file

    print(f"Parsing data: {dti_dir}")

    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")
    
    # Check required columns
    if drug_vec not in drug_df.columns:
        raise KeyError(f"Column '{drug_vec}' not found in drug data. Available: {drug_df.columns.tolist()}")
    
    if prot_vec != "Convolution" and prot_vec not in protein_df.columns:
        raise KeyError(f"Column '{prot_vec}' not found in protein data. Available: {protein_df.columns.tolist()}")
    
    if prot_vec == "Convolution" and "Sequence" not in protein_df.columns:
        raise KeyError("Column 'Sequence' required for Convolution mode but not found in protein data.")

    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    
    # Process drug features
    try:
        drug_feature = np.stack(dti_df[drug_vec].map(lambda fp: [float(x) for x in fp.split("\t")]))
    except:
        fp_array = dti_df[drug_vec].map(lambda fp: fp.split("\t"))
        first_fp = fp_array.iloc[0]
        if isinstance(first_fp, list) and len(first_fp) > 0:
            if not isinstance(first_fp[0], (float, int, np.float32, np.float64, np.int32, np.int64)):
                drug_feature = np.stack(fp_array.map(lambda fp_list: [float(x) for x in fp_list]))
            else:
                drug_feature = np.stack(fp_array)
        else:
            drug_feature = np.stack(dti_df[drug_vec].map(lambda fp: fp.split("\t")))
    
    if drug_feature.dtype.kind not in 'fiu':
        drug_feature = drug_feature.astype(np.float32)
        
    if prot_vec=="Convolution":
        protein_feature = sequence.pad_sequences(dti_df["encoded_sequence"].values, prot_len)
    else:
        protein_feature = np.stack(dti_df[prot_vec].map(lambda fp: fp.split("\t")))
        
    # Get drug and protein IDs
    drug_ids = dti_df[drug_col].tolist()
    protein_ids = dti_df[protein_col].tolist()
    
    # Load drug LM embeddings
    if drug_lm_file and os.path.exists(drug_lm_file):
        print(f"Loading drug embeddings from {drug_lm_file}")
        try:
            import torch
            embedding_dict = torch.load(drug_lm_file, weights_only=True)
            
            drug_embedding_tensor = embedding_dict.get('drug_embeddings', None)
            file_drug_ids = embedding_dict.get('drug_ids', [])
            
            if drug_embedding_tensor is None and isinstance(embedding_dict, dict):
                for key, value in embedding_dict.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                        drug_embedding_tensor = value
                        break
            
            if drug_embedding_tensor is not None and file_drug_ids:
                if drug_embedding_tensor.device.type != 'cpu':
                    drug_embedding_tensor = drug_embedding_tensor.cpu()
                
                drug_id_to_idx = {str(id_): idx for idx, id_ in enumerate(file_drug_ids)}
                
                actual_embedding_size = drug_embedding_tensor.shape[1]
                if actual_embedding_size != lm_embedding_size:
                    print(f"Adjusting embedding size from {lm_embedding_size} to {actual_embedding_size}")
                    lm_embedding_size = actual_embedding_size
                
                drug_lm_embeddings = np.zeros((len(drug_ids), lm_embedding_size), dtype=np.float32)
                
                missing_count = 0
                for i, drug_id in enumerate(drug_ids):
                    str_drug_id = str(drug_id)
                    if str_drug_id in drug_id_to_idx:
                        idx = drug_id_to_idx[str_drug_id]
                        drug_lm_embeddings[i] = drug_embedding_tensor[idx].numpy()
                    else:
                        missing_count += 1

                if missing_count > 0:
                    print(f"Warning: {missing_count}/{len(drug_ids)} drug IDs not found in embeddings")
            else:
                print("Warning: Drug embeddings not found, using zero vectors")
                drug_lm_embeddings = np.zeros((len(drug_ids), lm_embedding_size), dtype=np.float32)
        except Exception as e:
            print(f"Error loading drug embeddings: {str(e)}")
            drug_lm_embeddings = np.zeros((len(drug_ids), lm_embedding_size), dtype=np.float32)
    else:
        drug_lm_embeddings = np.zeros((len(drug_ids), lm_embedding_size), dtype=np.float32)
    
    # Load protein LM embeddings
    if protein_lm_file and os.path.exists(protein_lm_file):
        print(f"Loading protein embeddings from {protein_lm_file}")
        try:
            import torch
            embedding_dict = torch.load(protein_lm_file, weights_only=True)
            
            protein_embedding_tensor = embedding_dict.get('target_embeddings', None)
            file_protein_ids = embedding_dict.get('target_ids', [])
            
            if protein_embedding_tensor is None and isinstance(embedding_dict, dict):
                for key, value in embedding_dict.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 2 and key != 'drug_embeddings':
                        protein_embedding_tensor = value
                        break
            
            if protein_embedding_tensor is not None and file_protein_ids:
                if protein_embedding_tensor.device.type != 'cpu':
                    protein_embedding_tensor = protein_embedding_tensor.cpu()
                
                protein_id_to_idx = {str(id_): idx for idx, id_ in enumerate(file_protein_ids)}
                
                protein_lm_embeddings = np.zeros((len(protein_ids), lm_embedding_size), dtype=np.float32)
                
                missing_count = 0
                for i, protein_id in enumerate(protein_ids):
                    str_protein_id = str(protein_id)
                    if str_protein_id in protein_id_to_idx:
                        idx = protein_id_to_idx[str_protein_id]
                        protein_lm_embeddings[i] = protein_embedding_tensor[idx].numpy()
                    else:
                        missing_count += 1

                if missing_count > 0:
                    print(f"Warning: {missing_count}/{len(protein_ids)} protein IDs not found in embeddings")
            else:
                print("Warning: Protein embeddings not found, using zero vectors")
                protein_lm_embeddings = np.zeros((len(protein_ids), lm_embedding_size), dtype=np.float32)
        except Exception as e:
            print(f"Error loading protein embeddings: {str(e)}")
            protein_lm_embeddings = np.zeros((len(protein_ids), lm_embedding_size), dtype=np.float32)
    else:
        protein_lm_embeddings = np.zeros((len(protein_ids), lm_embedding_size), dtype=np.float32)
    
    print(f"Data shapes - Drug: {drug_feature.shape}, Protein: {protein_feature.shape}, Drug LM: {drug_lm_embeddings.shape}, Protein LM: {protein_lm_embeddings.shape}")
    
    if with_label:
        label = dti_df[label_col].values
        print(f"Labels - Positive: {sum(dti_df[label_col])}, Negative: {dti_df.shape[0] - sum(dti_df[label_col])}")
        return {"protein_feature": protein_feature, "drug_feature": drug_feature, 
                "drug_lm_embedding": drug_lm_embeddings, "protein_lm_embedding": protein_lm_embeddings,
                "label": label, "Compound_ID":dti_df["Compound_ID"].tolist(), "Protein_ID":dti_df["Protein_ID"].tolist()}
    else:
        return {"protein_feature": protein_feature, "drug_feature": drug_feature,
                "drug_lm_embedding": drug_lm_embeddings, "protein_lm_embedding": protein_lm_embeddings,
                "Compound_ID":dti_df["Compound_ID"].tolist(), "Protein_ID":dti_df["Protein_ID"].tolist()}



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", action="store_true", default=False)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp_r2")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    parser.add_argument("--lm-embedding-size", "-M", help="Size of LM embeddings vector", type=int, default=768)
    # LLM embedding files
    parser.add_argument("--drug-lm-file", help="File containing drug LLM embeddings (PT file with ID to embedding mapping)", default=None)
    parser.add_argument("--protein-lm-file", help="File containing protein LLM embeddings (PT file with ID to embedding mapping)", default=None)
    parser.add_argument("--embedding-file", help="Path to single .pt file containing both drug and protein embeddings", default=None)
    args = parser.parse_args()
    
    model = args.model

    test_names = args.test_name
    tests = args.test_dti_dir
    test_proteins = args.test_protein_dir
    test_drugs = args.test_drug_dir
    test_sets = zip(test_names, tests, test_drugs, test_proteins)
    with_label = args.with_label
    output_file = args.output

    print(f"Loading model: {model}")

    f = h5py.File(model, 'r+')

    try:
        f.__delitem__("optimizer_weights")
    except:
        pass

    f.close()

    type_params = {
        "prot_len": args.prot_len,
        "prot_vec": args.prot_vec,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
        "lm_embedding_size": args.lm_embedding_size,
        "drug_lm_file": args.drug_lm_file,
        "protein_lm_file": args.protein_lm_file,
        "embedding_file": args.embedding_file
    }
    
    if args.embedding_file:
        args.drug_lm_file = args.embedding_file
        args.protein_lm_file = args.embedding_file

    test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, with_label=with_label, **type_params)
                for test_name, test_dti, test_drug, test_protein in test_sets}

    loaded_model = load_model(model)
    
    print("Starting prediction...")
    result_df = pd.DataFrame()
    result_columns = []
    for dataset in test_dic:
        print(f"Predicting dataset: {dataset}")
        temp_df = pd.DataFrame()
        prediction_dic = test_dic[dataset]
        
        print(f"Dataset size: {prediction_dic['drug_feature'].shape[0]} samples")
        
        N = int(np.ceil(prediction_dic["drug_feature"].shape[0]/50))
        
        d_splitted = np.array_split(prediction_dic["drug_feature"], N)
        p_splitted = np.array_split(prediction_dic["protein_feature"], N)
        dlm_splitted = np.array_split(prediction_dic["drug_lm_embedding"], N)
        plm_splitted = np.array_split(prediction_dic["protein_lm_embedding"], N)
        
        predicted_batches = []
        for i, (d, p, dlm, plm) in enumerate(zip(d_splitted, p_splitted, dlm_splitted, plm_splitted)):
            batch_result = loaded_model.predict([d, p, dlm, plm], verbose=0)
            predicted_batches.append(np.squeeze(batch_result).tolist())
        
        predicted = sum(predicted_batches, [])
        
        predicted_array = np.array(predicted)
        print(f"Prediction range: [{predicted_array.min():.4f}, {predicted_array.max():.4f}], mean: {predicted_array.mean():.4f}")
        
        temp_df[dataset, 'predicted'] = predicted
        temp_df[dataset, 'Compound_ID'] = prediction_dic["Compound_ID"]
        temp_df[dataset, 'Protein_ID'] = prediction_dic["Protein_ID"]
        if with_label:
           temp_df[dataset, 'label'] = np.squeeze(test_dic[dataset]['label'])
        
        result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
        result_columns.append((dataset, "predicted"))
        result_columns.append((dataset, "Compound_ID"))
        result_columns.append((dataset, "Protein_ID"))
        if with_label:
           result_columns.append((dataset, "label"))
    
    result_df.columns = pd.MultiIndex.from_tuples(result_columns)
    print(f"Saving results to: {output_file}")
    result_df.to_csv(output_file, index=False)
    print("Prediction completed!")
