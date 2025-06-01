import numpy as np
import pandas as pd

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.preprocessing import sequence


from sklearn.metrics import precision_recall_curve, auc, roc_curve

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
               lm_embedding_size=1024, drug_lm_file=None, protein_lm_file=None, embedding_file=None):  

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
        
    drug_ids = dti_df[drug_col].tolist()
    protein_ids = dti_df[protein_col].tolist()
    
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
                "protein_lm_embedding": protein_lm_embeddings, "drug_lm_embedding": drug_lm_embeddings, 
                "label": label}
    else:
        return {"protein_feature": protein_feature, "drug_feature": drug_feature,
                "protein_lm_embedding": protein_lm_embeddings, "drug_lm_embedding": drug_lm_embeddings}


class Drug_Target_Prediction(object):
    
    def PLayer(self, size, filters, activation, initializer, regularizer_param):
        def f(input):
            model_p = Convolution1D(filters=filters, kernel_size=size, padding='same', kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f

    def modelv(self, dropout, drug_layers, protein_strides, filters, fc_layers, prot_vec=False, prot_len=2500,
               activation='relu', protein_layers=None, initializer="glorot_normal", drug_len=2048, drug_vec="morgan_fp_r2",
               lm_embedding_size=1024):  
        def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)

        regularizer_param = 0.001
        input_d = Input(shape=(drug_len,))
        input_p = Input(shape=(prot_len,))
        
        input_d_lm = Input(shape=(lm_embedding_size,))
        input_p_lm = Input(shape=(lm_embedding_size,))
        
        params_dic = {"kernel_initializer": initializer,
                      "kernel_regularizer": l2(regularizer_param),
        }
        input_layer_d = input_d
        if drug_layers is not None:
            drug_layers = return_tuple(drug_layers)
            for layer_size in drug_layers:
                model_d = Dense(layer_size, **params_dic)(input_layer_d)
                model_d = BatchNormalization()(model_d)
                model_d = Activation(activation)(model_d)
                model_d = Dropout(dropout)(model_d)
                input_layer_d = model_d

        if prot_vec == "Convolution":
            model_p = Embedding(26,20, embeddings_initializer=initializer,embeddings_regularizer=l2(regularizer_param))(input_p)
            model_p = SpatialDropout1D(0.2)(model_p)
            model_ps = [self.PLayer(stride_size, filters, activation, initializer, regularizer_param)(model_p) for stride_size in protein_strides]
            if len(model_ps)!=1:
                model_p = Concatenate(axis=1)(model_ps)
            else:
                model_p = model_ps[0]
        else:
            model_p = input_p

        if protein_layers:
            input_layer_p = model_p
            protein_layers = return_tuple(protein_layers)
            for protein_layer in protein_layers:
                model_p = Dense(protein_layer, **params_dic)(input_layer_p)
                model_p = BatchNormalization()(model_p)
                model_p = Activation(activation)(model_p)
                model_p = Dropout(dropout)(model_p)
                input_layer_p = model_p

        model_d = Concatenate(axis=1)([model_d, input_d_lm])
        model_p = Concatenate(axis=1)([model_p, input_p_lm])
        
        model_t = Concatenate(axis=1)([model_d,model_p])

        if fc_layers is not None:
            fc_layers = return_tuple(fc_layers)
            for fc_layer in fc_layers:
                model_t = Dense(units=fc_layer,
                                **params_dic)(model_t)
                model_t = BatchNormalization()(model_t)
                model_t = Activation(activation)(model_t)
                input_dim = fc_layer
        model_t = Dense(1, activation='tanh', activity_regularizer=l2(regularizer_param),**params_dic)(model_t)
        model_t = Lambda(lambda x: (x+1.)/2.)(model_t)

        model_f = Model(inputs=[input_d, input_p, input_d_lm, input_p_lm], outputs=model_t)

        return model_f

    def __init__(self, dropout=0.2, drug_layers=(1024,512), protein_windows = (10,15,20,25,30), filters=64,
                 learning_rate=1e-3, decay=0.0, fc_layers=None, prot_vec=None, prot_len=2500, activation="relu",
                 drug_len=2048, drug_vec="morgan_fp_r2", protein_layers=None, lm_embedding_size=1024):
        self.__dropout = dropout
        self.__drugs_layer = drug_layers
        self.__protein_strides = protein_windows
        self.__filters = filters
        self.__fc_layers = fc_layers
        self.__learning_rate = learning_rate
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__drug_vec = drug_vec
        self.__drug_len = drug_len
        self.__activation = activation
        self.__protein_layers = protein_layers
        self.__decay = decay
        self.__lm_embedding_size = lm_embedding_size
        self.__model_t = self.modelv(self.__dropout, self.__drugs_layer, self.__protein_strides,
                                     self.__filters, self.__fc_layers, prot_vec=self.__prot_vec,
                                     prot_len=self.__prot_len, activation=self.__activation,
                                     protein_layers=self.__protein_layers, drug_vec=self.__drug_vec,
                                     drug_len=self.__drug_len, lm_embedding_size=self.__lm_embedding_size)

        opt = Adam(lr=learning_rate, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        K.get_session().run(tf.global_variables_initializer())

    def fit(self, drug_feature, protein_feature, drug_lm_embedding, protein_lm_embedding, label, n_epoch=10, batch_size=32):
        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feature,protein_feature, drug_lm_embedding, protein_lm_embedding],label, epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1,initial_epoch=_)
        return self.__model_t
    
    def summary(self):
        self.__model_t.summary()
    
    def validation(self, drug_feature, protein_feature, drug_lm_embedding, protein_lm_embedding, label, output_file=None, n_epoch=10, batch_size=32, **kwargs):

        if output_file:
            param_tuple = pd.MultiIndex.from_tuples([("parameter", param) for param in ["window_sizes", "drug_layers", "fc_layers", "learning_rate"]])
            result_df = pd.DataFrame(data = [[self.__protein_strides, self.__drugs_layer, self.__fc_layers, self.__learning_rate]]*n_epoch, columns=param_tuple)
            result_df["epoch"] = range(1,n_epoch+1)
        result_dic = {dataset: {"AUC":[], "AUPR": [], "opt_threshold(AUPR)":[], "opt_threshold(AUC)":[] }for dataset in kwargs}

        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feature,protein_feature, drug_lm_embedding, protein_lm_embedding],label,
                                         epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1, initial_epoch=_)
            for dataset in kwargs:
                print(f"Validating {dataset}")
                test_p = kwargs[dataset]["protein_feature"]
                test_d = kwargs[dataset]["drug_feature"]
                test_drug_lm_embedding = kwargs[dataset]["drug_lm_embedding"]
                test_protein_lm_embedding = kwargs[dataset]["protein_lm_embedding"]
                test_label = kwargs[dataset]["label"]
                prediction = self.__model_t.predict([test_d,test_p, test_drug_lm_embedding, test_protein_lm_embedding])
                fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label,prediction)
                distance = (1-fpr)**2+(1-tpr)**2
                EERs = (1-recall)/(1-precision)
                positive = sum(test_label)
                negative = test_label.shape[0]-positive
                ratio = negative/positive
                opt_t_AUC = thresholds_AUC[np.argmin(distance)]
                opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-ratio))]
                AUPR = auc(recall,precision)
                print(f"AUC: {AUC:.3f}, AUPR: {AUPR:.3f}, Optimal threshold (AUC): {opt_t_AUC:.3f}, Optimal threshold (AUPR): {opt_t_AUPR:.3f}")
                result_dic[dataset]["AUC"].append(AUC)
                result_dic[dataset]["AUPR"].append(AUPR)
                result_dic[dataset]["opt_threshold(AUC)"].append(opt_t_AUC)
                result_dic[dataset]["opt_threshold(AUPR)"].append(opt_t_AUPR)
        if output_file:
            for dataset in kwargs:
                result_df[dataset, "AUC"] = result_dic[dataset]["AUC"]
                result_df[dataset, "AUPR"] = result_dic[dataset]["AUPR"]
                result_df[dataset, "opt_threshold(AUC)"] = result_dic[dataset]["opt_threshold(AUC)"]
                result_df[dataset, "opt_threshold(AUPR)"] = result_dic[dataset]["opt_threshold(AUPR)"]
            print(f"Saving results to {output_file}")
            result_df.to_csv(output_file, index=False)

    def predict(self, **kwargs):
        results_dic = {}
        for dataset in kwargs:
            result_dic = {}
            test_p = kwargs[dataset]["protein_feature"]
            test_d = kwargs[dataset]["drug_feature"]
            test_drug_lm_embedding = kwargs[dataset]["drug_lm_embedding"]
            test_protein_lm_embedding = kwargs[dataset]["protein_lm_embedding"]
            result_dic["label"] = kwargs[dataset]["label"]
            result_dic["predicted"] = self.__model_t.predict([test_d, test_p, test_drug_lm_embedding, test_protein_lm_embedding])
            results_dic[dataset] = result_dic
        return results_dic
    
    def save(self, output_file):
        self.__model_t.save(output_file)


if __name__ == '__main__':
    import argparse
    import os  

    parser = argparse.ArgumentParser()
    # train_params
    parser.add_argument("dti_dir", help="Training DTI information [drug, target, label]")
    parser.add_argument("drug_dir", help="Training drug information [drug, SMILES,[feature_name, ..]]")
    parser.add_argument("protein_dir", help="Training protein information [protein, seq, [feature_name]]")
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", action="store_true")
    # structure_params
    parser.add_argument("--window-sizes", '-w', help="Window sizes for model (only works for Convolution)", default=[10, 15, 20, 25, 30], nargs="*", type=int)
    parser.add_argument("--protein-layers","-p", help="Dense layers for protein", default=[128, 64], nargs="*", type=int)
    parser.add_argument("--drug-layers", '-c', help="Dense layers for drugs", default=[128], nargs="*", type=int)
    parser.add_argument("--fc-layers", '-f', help="Dense layers for concatenated layers of drug and target layer", default=[256], nargs="*", type=int)
    # training_params
    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="The number of epochs for training or validation", type=int, default=15)
    # type_params
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution") 
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp_r2")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    parser.add_argument("--lm-embedding-size", "-M", help="Size of LLM embedding", type=int, default=1024)  
    parser.add_argument("--drug-lm-file", dest="drug_lm_file", help="Path to LM embeddings for drugs (.pt file)", default=None)
    parser.add_argument("--protein-lm-file", dest="protein_lm_file", help="Path to LM embeddings for proteins (.pt file)", default=None)
    parser.add_argument("--embedding-file", dest="embedding_file", help="Path to single .pt file containing both drug and protein embeddings", default=None)
    # the other hyper-parameters
    parser.add_argument("--activation", "-a", help='Activation function of model', type=str, default='elu')
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
    parser.add_argument("--n-filters", "-F", help="Number of filters for convolution layer, only works for Convolution", default=64, type=int)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
    parser.add_argument("--decay", "-y", help="Learning rate decay", default=1e-4, type=float)
    # mode_params
    parser.add_argument("--validation", help="Excute validation with independent data, will give AUC and AUPR (No prediction result)", action="store_true", default=False)
    parser.add_argument("--predict", help="Predict interactions of independent test set", action="store_true", default=False)
    # output_params
    parser.add_argument("--save-model", "-m", help="save model", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)

    args = parser.parse_args()
    # train data
    train_dic = {
        "dti_dir": args.dti_dir,
        "drug_dir": args.drug_dir,
        "protein_dir": args.protein_dir,
        "with_label": True
    }
    test_names = args.test_name
    tests = args.test_dti_dir
    test_proteins = args.test_protein_dir
    test_drugs = args.test_drug_dir
    if test_names is None:
        test_sets = []
    else:
        test_sets = list(zip(test_names, tests, test_drugs, test_proteins))

    valid_sets = test_sets.copy() if test_sets else []
    
    output_file = args.output
    
    model_params = {
        "drug_layers": args.drug_layers,
        "protein_windows": args.window_sizes,
        "protein_layers": args.protein_layers,
        "fc_layers": args.fc_layers,
        "learning_rate": args.learning_rate,
        "decay": args.decay,
        "activation": args.activation,
        "filters": args.n_filters,
        "dropout": args.dropout,
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
        "lm_embedding_size": args.lm_embedding_size
    }
    
    data_params = {
        "drug_lm_file": args.drug_lm_file,
        "protein_lm_file": args.protein_lm_file,
        "embedding_file": args.embedding_file
    }
    print("Model parameters:")
    for key in model_params.keys():
        print(f"  {key}: {model_params[key]}")

    if args.embedding_file:
        print(f"Using embedding file: {args.embedding_file}")
        args.drug_lm_file = args.embedding_file
        args.protein_lm_file = args.embedding_file
    
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
    
    train_paths = {
        "dti_dir": args.dti_dir,
        "drug_dir": args.drug_dir,
        "protein_dir": args.protein_dir
    }
    train_dic.update(type_params)
    train_dic = parse_data(**train_dic)
    test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, with_label=True, **type_params)
                for test_name, test_dti, test_drug, test_protein in test_sets}

    if args.predict:
        print("Starting prediction mode")
        train_dic.update({"n_epoch": args.n_epoch, "batch_size": args.batch_size})
        dti_prediction_model = Drug_Target_Prediction(**model_params)
        dti_prediction_model.fit(train_dic["drug_feature"], train_dic["protein_feature"], train_dic["drug_lm_embedding"], train_dic["protein_lm_embedding"], train_dic["label"], **train_dic)
        test_predicted = dti_prediction_model.predict(**test_dic)
        result_df = pd.DataFrame()
        result_columns = []
        for dataset in test_predicted:
            temp_df = pd.DataFrame()
            value = test_predicted[dataset]["predicted"]
            value = np.squeeze(value)
            temp_df[dataset,'predicted'] = value
            temp_df[dataset, 'label'] = np.squeeze(test_predicted[dataset]['label'])
            result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
            result_columns.append((dataset, "predicted"))
            result_columns.append((dataset, "label"))
        result_df.columns = pd.MultiIndex.from_tuples(result_columns)
        print(f"Saving results to {output_file}")
        result_df.to_csv(output_file, index=False)

    if args.validation:
        print("Starting validation mode")
        output_file = args.output
        validation_params = {
            "n_epoch": args.n_epoch,
            "batch_size": args.batch_size,
            "output_file": output_file,
        }

        if args.embedding_file:
            if not os.path.exists(args.embedding_file):
                print(f"Warning: Embedding file {args.embedding_file} does not exist!")
            args.drug_lm_file = args.embedding_file
            args.protein_lm_file = args.embedding_file

        data_params = {
            "prot_len": args.prot_len,
            "prot_vec": args.prot_vec,
            "drug_vec": args.drug_vec,
            "drug_len": args.drug_len,
            "lm_embedding_size": args.lm_embedding_size,
            "drug_lm_file": args.drug_lm_file,
            "protein_lm_file": args.protein_lm_file,
        }

        train_data = train_dic

        if len(valid_sets) == 0:
            print("Warning: No validation sets defined!")
        
        valid_data = {}
        for valid_name, valid_dti, valid_drug, valid_protein in valid_sets:
            print(f"Processing validation set: {valid_name}")

            if valid_name in test_dic:
                valid_data[valid_name] = test_dic[valid_name]
            else:
                valid_params = {
                    "dti_dir": valid_dti,
                    "drug_dir": valid_drug, 
                    "protein_dir": valid_protein,
                    "with_label": True
                }
                valid_params.update(data_params)
                valid_data[valid_name] = parse_data(**valid_params)

        print("Creating DTI prediction model")
        dti_prediction_model = Drug_Target_Prediction(**model_params)

        print("Starting validation")
        validation_params.update(valid_data)
        dti_prediction_model.validation(train_data["drug_feature"], train_data["protein_feature"], 
                                       train_data["drug_lm_embedding"], train_data["protein_lm_embedding"], 
                                       train_data["label"], **validation_params)

    if args.save_model:
        print(f"Saving model to {args.save_model}")
        dti_prediction_model.save(args.save_model)
    exit()
