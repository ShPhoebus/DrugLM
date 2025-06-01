import numpy as np
import pandas as pd
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

def parse_data(dti_dir, drug_dir, protein_dir, with_label=True, lm_embedding_size=1024, embedding_file=None):  
    print(f"Parsing files {dti_dir}, {drug_dir}, {protein_dir}")
    print(f"Using embedding file: {embedding_file}, embedding size: {lm_embedding_size}")

    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")
    
    print("DTI data sample:", dti_df.head())
    
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    
    drug_ids = dti_df[drug_col].tolist()
    protein_ids = dti_df[protein_col].tolist()
    
    drug_lm_embeddings = np.zeros((len(drug_ids), lm_embedding_size), dtype=np.float32)
    protein_lm_embeddings = np.zeros((len(protein_ids), lm_embedding_size), dtype=np.float32)
    
    if embedding_file and os.path.exists(embedding_file):
        print(f"Loading embeddings from {embedding_file}")
        try:
            import torch
            embedding_dict = torch.load(embedding_file, weights_only=True)
            
            drug_embedding_tensor = embedding_dict.get('drug_embeddings', None)
            file_drug_ids = embedding_dict.get('drug_ids', [])
            
            protein_embedding_tensor = embedding_dict.get('target_embeddings', None)
            file_protein_ids = embedding_dict.get('target_ids', [])
            
            if drug_embedding_tensor is None and isinstance(embedding_dict, dict):
                for key, value in embedding_dict.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                        if key != 'target_embeddings' and drug_embedding_tensor is None:
                            print(f"Found potential drug embeddings: {key}, shape: {value.shape}")
                            drug_embedding_tensor = value
                            continue
                        if key != 'drug_embeddings' and protein_embedding_tensor is None:
                            print(f"Found potential protein embeddings: {key}, shape: {value.shape}")
                            protein_embedding_tensor = value
            
            if drug_embedding_tensor is not None and file_drug_ids:
                if drug_embedding_tensor.device.type != 'cpu':
                    drug_embedding_tensor = drug_embedding_tensor.cpu()
                
                drug_id_to_idx = {str(id_): idx for idx, id_ in enumerate(file_drug_ids)}
                print(f"Loaded {len(file_drug_ids)} drug embeddings with dimension {drug_embedding_tensor.shape[1]}")
                
                missing_drugs = []
                for i, drug_id in enumerate(drug_ids):
                    str_drug_id = str(drug_id)
                    if str_drug_id in drug_id_to_idx:
                        idx = drug_id_to_idx[str_drug_id]
                        drug_lm_embeddings[i] = drug_embedding_tensor[idx].numpy()
                    else:
                        missing_drugs.append(str_drug_id)
                
                if missing_drugs:
                    print(f"Warning: {len(missing_drugs)}/{len(drug_ids)} drug IDs not found in embedding file")
                    if len(missing_drugs) <= 10:
                        print(f"Missing IDs: {missing_drugs}")
                    else:
                        print(f"First 10 missing IDs: {missing_drugs[:10]}...")
            
            if protein_embedding_tensor is not None and file_protein_ids:
                if protein_embedding_tensor.device.type != 'cpu':
                    protein_embedding_tensor = protein_embedding_tensor.cpu()
                
                protein_id_to_idx = {str(id_): idx for idx, id_ in enumerate(file_protein_ids)}
                print(f"Loaded {len(file_protein_ids)} protein embeddings with dimension {protein_embedding_tensor.shape[1]}")
                
                missing_proteins = []
                for i, protein_id in enumerate(protein_ids):
                    str_protein_id = str(protein_id)
                    if str_protein_id in protein_id_to_idx:
                        idx = protein_id_to_idx[str_protein_id]
                        protein_lm_embeddings[i] = protein_embedding_tensor[idx].numpy()
                    else:
                        missing_proteins.append(str_protein_id)
                
                if missing_proteins:
                    print(f"Warning: {len(missing_proteins)}/{len(protein_ids)} protein IDs not found in embedding file")
                    if len(missing_proteins) <= 10:
                        print(f"Missing IDs: {missing_proteins}")
                    else:
                        print(f"First 10 missing IDs: {missing_proteins[:10]}...")
        
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            print("Using zero vectors as embeddings due to error")
    else:
        print("No embedding file provided or file doesn't exist, using zero vectors as embeddings")
    
    print("\nSample drug embeddings:")
    for i in range(min(3, len(drug_ids))):
        print(f"Drug ID: {drug_ids[i]}")
        print(f"LM embedding (first 5 values): {drug_lm_embeddings[i][:5]}")
    
    print("\nSample protein embeddings:")
    for i in range(min(3, len(protein_ids))):
        print(f"Protein ID: {protein_ids[i]}")
        print(f"LM embedding (first 5 values): {protein_lm_embeddings[i][:5]}")
    
    if with_label:
        label = dti_df[label_col].values
        print(f"\tPositive samples: {sum(dti_df[label_col])}")
        print(f"\tNegative samples: {dti_df.shape[0] - sum(dti_df[label_col])}")
        return {
            "drug_lm_embedding": drug_lm_embeddings, 
            "protein_lm_embedding": protein_lm_embeddings, 
            "label": label,
            "drug_ids": drug_ids,
            "protein_ids": protein_ids
        }
    else:
        return {
            "drug_lm_embedding": drug_lm_embeddings, 
            "protein_lm_embedding": protein_lm_embeddings,
            "drug_ids": drug_ids,
            "protein_ids": protein_ids
        }


class MLP_DTI_Model:
    def __init__(self, lm_embedding_size=1024, fc_layers=(512, 256, 128), 
                 dropout=0.2, learning_rate=1e-4, decay=1e-4, activation='elu'):
        self.lm_embedding_size = lm_embedding_size
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.decay = decay
        self.activation = activation
        
        self.model = self._build_model()
        
        optimizer = Adam(lr=learning_rate, decay=decay)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
    def _build_model(self):
        regularizer_param = 0.001
        
        input_d_lm = Input(shape=(self.lm_embedding_size,))
        input_p_lm = Input(shape=(self.lm_embedding_size,))
        
        merged = Concatenate(axis=1)([input_d_lm, input_p_lm])
        
        x = merged
        for units in self.fc_layers:
            x = Dense(units, kernel_initializer='glorot_normal', 
                     kernel_regularizer=l2(regularizer_param))(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            x = Dropout(self.dropout)(x)
        
        output = Dense(1, activation='tanh', kernel_initializer='glorot_normal',
                      kernel_regularizer=l2(regularizer_param))(x)
        output = tf.keras.layers.Lambda(lambda x: (x+1.)/2.)(output)
        
        model = Model(inputs=[input_d_lm, input_p_lm], outputs=output)
        
        return model
    
    def save_embeddings(self, train_data, path='trained_embeddings', with_id_mapping=True):
        print(f"\nSaving trained node embeddings")
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        drug_embeddings = train_data['drug_lm_embedding']
        protein_embeddings = train_data['protein_lm_embedding']
        
        np.save(os.path.join(path, 'user_embeddings.npy'), drug_embeddings)
        np.save(os.path.join(path, 'item_embeddings.npy'), protein_embeddings)
        
        drug_input = self.model.get_layer(index=0).output
        protein_input = self.model.get_layer(index=1).output
        
        concat_layer = None
        for layer in self.model.layers:
            if isinstance(layer, Concatenate):
                concat_layer = layer.output
                break
        
        if concat_layer is not None:
            intermediate_model = Model(inputs=[drug_input, protein_input], outputs=concat_layer)
            
            concat_embeddings = intermediate_model.predict([drug_embeddings, protein_embeddings])
            
            drug_processed = concat_embeddings[:, :self.lm_embedding_size]
            protein_processed = concat_embeddings[:, self.lm_embedding_size:]
            
            np.save(os.path.join(path, 'user_gcn_embeddings.npy'), drug_processed)
            np.save(os.path.join(path, 'item_gcn_embeddings.npy'), protein_processed)
        else:
            np.save(os.path.join(path, 'user_gcn_embeddings.npy'), drug_embeddings)
            np.save(os.path.join(path, 'item_gcn_embeddings.npy'), protein_embeddings)
        
        if with_id_mapping:
            try:
                drug_ids = train_data['drug_ids']
                protein_ids = train_data['protein_ids']
                
                unique_drug_ids = list(set(drug_ids))
                unique_protein_ids = list(set(protein_ids))
                
                drug_id_mapping = {str(drug_id): idx for idx, drug_id in enumerate(unique_drug_ids)}
                protein_id_mapping = {str(protein_id): idx for idx, protein_id in enumerate(unique_protein_ids)}
                
                id_mappings = {
                    'drug': drug_id_mapping,
                    'target': protein_id_mapping
                }
                
                with open(os.path.join(path, 'id_mappings.json'), 'w') as f:
                    json.dump(id_mappings, f, indent=4)
                    
            except Exception as e:
                print(f"Error saving ID mappings: {e}")
        
        print(f"Embeddings saved to {path} directory")
        print(f"User embedding shape: {drug_embeddings.shape}")
        print(f"Item embedding shape: {protein_embeddings.shape}")
        if concat_layer is not None:
            print(f"Processed user embedding shape: {drug_processed.shape}")
            print(f"Processed item embedding shape: {protein_processed.shape}")
    
    def fit(self, drug_lm_embedding, protein_lm_embedding, label, 
            n_epoch=10, batch_size=32, validation_data=None, verbose=1,
            save_emb_epochs=None, train_data=None, out_dir="./"):
        val_data = None
        if validation_data is not None:
            val_data = ([validation_data[0], validation_data[1]], validation_data[2])
        
        class EmbeddingSaveCallback(tf.keras.callbacks.Callback):
            def __init__(self, model_instance, save_epochs, train_data, out_dir):
                super().__init__()
                self.model_instance = model_instance
                self.save_epochs = save_epochs or []
                self.train_data = train_data
                self.out_dir = out_dir
            
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) in self.save_epochs:
                    print(f'\nSaving node embeddings for epoch {epoch + 1}...')
                    embeddings_path = os.path.join(self.out_dir, f'epoch_{epoch + 1}_embeddings')
                    if self.train_data:
                        self.model_instance.save_embeddings(self.train_data, path=embeddings_path)
                        print(f'Epoch {epoch + 1} embeddings saved to: {embeddings_path}')
                    else:
                        print("Training data not available, skipping embedding save")
        
        callbacks = []
        if save_emb_epochs and train_data:
            callbacks.append(EmbeddingSaveCallback(self, save_emb_epochs, train_data, out_dir))
            
        history = self.model.fit(
            [drug_lm_embedding, protein_lm_embedding], 
            label, 
            epochs=n_epoch, 
            batch_size=batch_size, 
            validation_data=val_data,
            verbose=verbose,
            shuffle=True,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, drug_lm_embedding, protein_lm_embedding):
        return self.model.predict([drug_lm_embedding, protein_lm_embedding])
    
    @classmethod
    def load(cls, filepath):
        model_instance = cls(lm_embedding_size=1)
        model_instance.model = load_model(filepath)
        return model_instance
    
    def summary(self):
        self.model.summary()
    
    def validate(self, drug_lm_embedding, protein_lm_embedding, label, output_file=None, dataset_name="validation"):
        prediction = self.predict(drug_lm_embedding, protein_lm_embedding)
        
        fpr, tpr, thresholds_AUC = roc_curve(label, prediction)
        AUC = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(label, prediction)
        AUPR = auc(recall, precision)
        
        distance = (1-fpr)**2 + (1-tpr)**2
        EERs = (1-recall)/(1-precision)
        positive = sum(label)
        negative = len(label) - positive
        ratio = negative/positive
        
        opt_t_AUC = thresholds_AUC[np.argmin(distance)]
        opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-ratio))]
        
        print(f"{dataset_name} Validation Results:")
        print(f"AUC: {AUC:.4f}")
        print(f"AUPR: {AUPR:.4f}")
        print(f"Optimal threshold (AUC): {opt_t_AUC:.4f}")
        print(f"Optimal threshold (AUPR): {opt_t_AUPR:.4f}")
        
        if output_file:
            result_df = pd.DataFrame({
                'prediction': prediction.flatten(),
                'label': label
            })
            result_df.to_csv(output_file, index=False)
            print(f"Saved prediction results to {output_file}")
        
        return {
            "AUC": AUC,
            "AUPR": AUPR,
            "optimal_threshold_AUC": opt_t_AUC,
            "optimal_threshold_AUPR": opt_t_AUPR,
            "prediction": prediction,
            "label": label
        }
    
    def evaluate_with_threshold(self, drug_lm_embedding, protein_lm_embedding, label, threshold, dataset_name="Test set"):
        prediction = self.predict(drug_lm_embedding, protein_lm_embedding)
        
        binary_pred = (prediction > threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(label, binary_pred).ravel()
        
        acc = accuracy_score(label, binary_pred)
        precision = precision_score(label, binary_pred)
        recall = recall_score(label, binary_pred)
        f1 = f1_score(label, binary_pred)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fpr, tpr, _ = roc_curve(label, prediction)
        AUC = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(label, prediction)
        AUPR = auc(rec, prec)
        
        print(f"{dataset_name} Performance Evaluation (threshold={threshold:.4f}):")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall/Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {AUC:.4f}")
        print(f"AUPR: {AUPR:.4f}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "AUC": AUC,
            "AUPR": AUPR,
            "prediction": prediction,
            "binary_prediction": binary_pred,
            "label": label,
            "confusion_matrix": (tn, fp, fn, tp)
        }


def main():
    parser = argparse.ArgumentParser(description='MLP-based DTI prediction model using only LLM embeddings')
    
    parser.add_argument('train_dti', help='Training set DTI file path')
    parser.add_argument('train_drug', help='Training set drug information file path')
    parser.add_argument('train_protein', help='Training set protein information file path')
    parser.add_argument('--valid-dti', help='Validation set DTI file path')
    parser.add_argument('--valid-drug', help='Validation set drug information file path')
    parser.add_argument('--valid-protein', help='Validation set protein information file path')
    parser.add_argument('--test-dti', help='Test set DTI file path')
    parser.add_argument('--test-drug', help='Test set drug information file path')
    parser.add_argument('--test-protein', help='Test set protein information file path')
    
    parser.add_argument('--embedding-file', help='PT file path containing drug and protein embeddings', required=True)
    parser.add_argument('--lm-embedding-size', type=int, default=1024, help='LLM embedding dimension')
    
    parser.add_argument('--fc-layers', type=int, nargs='+', default=[512, 256, 128], help='Fully connected layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-4, help='Learning rate decay')
    parser.add_argument('--activation', type=str, default='elu', help='Activation function')
    
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.35, help='Classification threshold')
    
    parser.add_argument('--save-emb-epochs', type=str, default="", 
                        help="Specify epochs to save embeddings, separated by commas, e.g. '10,20,30'")
    parser.add_argument('--out-dir', type=str, default="./", 
                        help="Output directory for saving embedding files")
    
    args = parser.parse_args()
    
    save_emb_epochs = []
    if args.save_emb_epochs:
        try:
            save_emb_epochs = [int(e) for e in args.save_emb_epochs.split(',')]
            print(f"Will save embeddings at epochs: {save_emb_epochs}")
        except:
            print("Error parsing save_emb_epochs parameter, format should be comma-separated numbers, e.g. '10,20,30'")
    
    print("Parameter Settings:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print()
    
    print("Loading Training Data")
    train_data = parse_data(
        args.train_dti, 
        args.train_drug, 
        args.train_protein, 
        with_label=True,
        lm_embedding_size=args.lm_embedding_size,
        embedding_file=args.embedding_file
    )
    
    valid_data = None
    if args.valid_dti and args.valid_drug and args.valid_protein:
        print("\nLoading Validation Data")
        valid_data = parse_data(
            args.valid_dti, 
            args.valid_drug, 
            args.valid_protein, 
            with_label=True,
            lm_embedding_size=args.lm_embedding_size,
            embedding_file=args.embedding_file
        )
    
    print("\nCreating Model")
    model = MLP_DTI_Model(
        lm_embedding_size=args.lm_embedding_size,
        fc_layers=args.fc_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        decay=args.decay,
        activation=args.activation
    )
    
    model.summary()
    
    print("\nTraining Model")
    validation_data = None
    if valid_data:
        validation_data = (
            valid_data['drug_lm_embedding'], 
            valid_data['protein_lm_embedding'], 
            valid_data['label']
        )
    
    history = model.fit(
        train_data['drug_lm_embedding'],
        train_data['protein_lm_embedding'],
        train_data['label'],
        n_epoch=args.epochs,
        batch_size=args.batch_size,
        validation_data=validation_data,
        save_emb_epochs=save_emb_epochs,
        train_data=train_data,
        out_dir=args.out_dir
    )
    
    if save_emb_epochs or True:
        print('\nSaving final trained node embeddings...')
        embeddings_path = os.path.join(args.out_dir, 'final_embeddings')
        model.save_embeddings(train_data, path=embeddings_path)
        print(f'Final embeddings saved to: {embeddings_path}')
    
    if valid_data:
        print("\nValidating Model on Validation Set")
        valid_output_file = "valid_result.csv"
        valid_results = model.validate(
            valid_data['drug_lm_embedding'],
            valid_data['protein_lm_embedding'],
            valid_data['label'],
            output_file=valid_output_file,
            dataset_name="Validation Set"
        )
        
        optimal_threshold = valid_results['optimal_threshold_AUPR']
        print(f"\nOptimal threshold based on validation set: {optimal_threshold:.4f}")
    else:
        optimal_threshold = args.threshold
        print(f"\nUsing threshold from command line arguments: {optimal_threshold:.4f}")
    
    if args.test_dti and args.test_drug and args.test_protein:
        print("\nLoading Test Data")
        test_data = parse_data(
            args.test_dti, 
            args.test_drug, 
            args.test_protein, 
            with_label=True,
            lm_embedding_size=args.lm_embedding_size,
            embedding_file=args.embedding_file
        )
        
        print("\nEvaluating Model on Test Set")
        test_results = model.evaluate_with_threshold(
            test_data['drug_lm_embedding'],
            test_data['protein_lm_embedding'],
            test_data['label'],
            threshold=optimal_threshold,
            dataset_name="Test Set"
        )
        
        test_output_file = "test_result.csv"
        test_df = pd.DataFrame({
            'drug_id': test_data['drug_ids'],
            'protein_id': test_data['protein_ids'],
            'prediction': test_results['prediction'].flatten(),
            'binary_prediction': test_results['binary_prediction'].flatten(),
            'label': test_results['label']
        })
        test_df.to_csv(test_output_file, index=False)
        print(f"Saved test set prediction results to {test_output_file}")
    
    print("\nCompleted")

if __name__ == '__main__':
    main()
