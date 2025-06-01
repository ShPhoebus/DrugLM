#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_mlp_dti(args):
    """Run MLP-based DTI prediction task"""
    print("Running MLP-based DTI prediction...")

    train_dti = "MLP_DTI/mydata/train_dti.csv"
    train_drug = "MLP_DTI/mydata/train_drug_information.csv"
    train_protein = "MLP_DTI/mydata/train_protein_information.csv"
    valid_dti = "MLP_DTI/mydata/valid_dti_filtered.csv"
    valid_drug = "MLP_DTI/mydata/valid_drug_information_filtered.csv"
    valid_protein = "MLP_DTI/mydata/valid_protein_information_filtered.csv"
    test_dti = "MLP_DTI/mydata/test_dti_filtered.csv"
    test_drug = "MLP_DTI/mydata/test_drug_information_filtered.csv"
    test_protein = "MLP_DTI/mydata/test_protein_information_filtered.csv"

    required_files = [
        train_dti, train_drug, train_protein,
        valid_dti, valid_drug, valid_protein,
        test_dti, test_drug, test_protein
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required data files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please ensure all data files are present in the MLP_DTI/mydata/ directory.")
        return False

    embedding_file = args.embedding_file
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file not found: {embedding_file}")
        return False

    cmd = [
        "python", "MLP_DTI/mlp_dti.py",
        train_dti, train_drug, train_protein,
        "--valid-dti", valid_dti,
        "--valid-drug", valid_drug,
        "--valid-protein", valid_protein,
        "--test-dti", test_dti,
        "--test-drug", test_drug,
        "--test-protein", test_protein,
        "--embedding-file", embedding_file,
        "--lm-embedding-size", str(args.lm_embedding_size),
        "--fc-layers"] + [str(x) for x in args.fc_layers] + [
        "--learning-rate", str(args.learning_rate),
        "--decay", str(args.decay),
        "--activation", args.activation,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size)
    ]

    if args.save_emb_epochs:
        cmd.extend(["--save-emb-epochs", args.save_emb_epochs])
    
    if args.out_dir:
        cmd.extend(["--out-dir", args.out_dir])

    try:
        result = subprocess.run(cmd, check=True)
        print("MLP DTI prediction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: MLP DTI prediction failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: Could not find MLP_DTI/mlp_dti.py")
        print("Please ensure the MLP_DTI directory and mlp_dti.py file exist.")
        return False

def run_gnn_dti(args):
    print(f"Running GNN-based DTI prediction ({args.gnn_model.upper()})...")

    embedding_file = args.embedding_file
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file not found: {embedding_file}")
        return False

    if not os.path.exists("GNN_DTI"):
        print("Error: GNN_DTI directory not found")
        return False
    
    if not os.path.exists("GNN_DTI/main.py"):
        print("Error: GNN_DTI/main.py not found")
        return False

    cmd = [
        "python", "GNN_DTI/main.py",
        "--dataset", "drug",
        "--epoch", str(args.epochs),
        "--gnn", args.gnn_model,
        "--dim", str(args.dim),
        "--lr", "0.001",
        "--batch_size", "1024",
        "--gpu_id", "0",
        "--context_hops", "3",
        "--pool", "mean",
        "--ns", "rns",
        "--K", "1",
        "--n_negs", "1",
        "--embedding-file", embedding_file
    ]

    original_dir = os.getcwd()
    try:
        os.chdir("GNN_DTI")

        cmd_adjusted = [
            "python", "main.py",
            "--dataset", "drug",
            "--epoch", str(args.epochs),
            "--gnn", args.gnn_model,
            "--dim", str(args.dim),
            "--lr", "0.001",
            "--batch_size", "1024",
            "--gpu_id", "0",
            "--context_hops", "3",
            "--pool", "mean",
            "--ns", "rns",
            "--K", "1",
            "--n_negs", "1",
            "--embedding-file", os.path.join("..", embedding_file)
        ]
        
        result = subprocess.run(cmd_adjusted, check=True)
        print(f"GNN DTI prediction ({args.gnn_model.upper()}) completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: GNN DTI prediction failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: Could not find GNN_DTI/main.py")
        return False
    finally:
        os.chdir(original_dir)

def run_deepconv_dti(args):
    print("Running DeepConvDTI-based DTI prediction...")

    train_dti = "DeepConvDTI/mydata/train_dti.csv"
    train_drug = "DeepConvDTI/mydata/train_drug_information.csv"
    train_protein = "DeepConvDTI/mydata/train_protein_information.csv"
    valid_dti = "DeepConvDTI/mydata/valid_dti_filtered.csv"
    valid_drug = "DeepConvDTI/mydata/valid_drug_information_filtered.csv"
    valid_protein = "DeepConvDTI/mydata/valid_protein_information_filtered.csv"
    test_dti = "DeepConvDTI/mydata/test_dti_filtered.csv"
    test_drug = "DeepConvDTI/mydata/test_drug_information_filtered.csv"
    test_protein = "DeepConvDTI/mydata/test_protein_information_filtered.csv"
    
    # Check if data files exist
    required_files = [
        train_dti, train_drug, train_protein,
        valid_dti, valid_drug, valid_protein,
        test_dti, test_drug, test_protein
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required data files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please ensure all data files are present in the DeepConvDTI/mydata/ directory.")
        return False

    embedding_file = args.embedding_file
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file not found: {embedding_file}")
        return False

    if not os.path.exists("DeepConvDTI"):
        print("Error: DeepConvDTI directory not found")
        return False
    
    if not os.path.exists("DeepConvDTI/DeepConvDTI.py"):
        print("Error: DeepConvDTI/DeepConvDTI.py not found")
        return False

    model_file = "DeepConvDTI/model_filtered.model"
    validation_output = "DeepConvDTI/validation_filtered_output.csv"
    test_output = "DeepConvDTI/test_filtered_result.csv"

    print("Training model...")
    train_cmd = [
        "python", "DeepConvDTI/DeepConvDTI.py",
        train_dti, train_drug, train_protein,
        "--validation",
        "-n", "validation_filtered",
        "-i", valid_dti,
        "-d", valid_drug,
        "-t", valid_protein,
        "-W",
        "-c"] + [str(x) for x in args.drug_layers] + [
        "-w"] + [str(x) for x in args.window_sizes] + [
        "-p"] + [str(x) for x in args.protein_layers] + [
        "-f"] + [str(x) for x in args.fc_layers] + [
        "-r", str(args.learning_rate),
        "-e", str(args.epochs),
        "-v", args.prot_vec,
        "-l", str(args.prot_len),
        "-V", args.drug_vec,
        "-L", str(args.drug_len),
        "-D", str(args.dropout),
        "-a", args.activation,
        "-F", str(args.n_filters),
        "-b", str(args.batch_size),
        "-y", str(args.decay),
        "-o", validation_output,
        "-m", model_file,
        "--embedding-file", embedding_file,
        "--lm-embedding-size", str(args.dim)
    ]
    
    try:
        result = subprocess.run(train_cmd, check=True)
        print("Training completed")
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: Could not find DeepConvDTI/DeepConvDTI.py")
        return False

    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} was not created during training")
        return False

    print("Predicting on test set...")
    predict_cmd = [
        "python", "DeepConvDTI/predict_with_model.py",
        model_file,
        "-n", "predict",
        "-i", test_dti,
        "-d", test_drug,
        "-t", test_protein,
        "-v", args.prot_vec,
        "-l", str(args.prot_len),
        "-V", args.drug_vec,
        "-L", str(args.drug_len),
        "-W",
        "-o", test_output,
        "--embedding-file", embedding_file,
        "--lm-embedding-size", str(args.dim)
    ]
    
    try:
        result = subprocess.run(predict_cmd, check=True)
        print("Prediction completed")
    except subprocess.CalledProcessError as e:
        print(f"Error: Prediction failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: Could not find DeepConvDTI/predict_with_model.py")
        return False

    if not os.path.exists(test_output):
        print(f"Error: Prediction file {test_output} was not created")
        return False

    if hasattr(args, 'threshold') and args.threshold is not None:
        print("Evaluating performance...")
        eval_cmd = [
            "python", "DeepConvDTI/evaluate_performance.py",
            test_output,
            "-n", "predict",
            "-T", str(args.threshold)
        ]
        
        try:
            result = subprocess.run(eval_cmd, check=True)
            print("Evaluation completed")
        except subprocess.CalledProcessError as e:
            print(f"Error: Evaluation failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print("Error: Could not find DeepConvDTI/evaluate_performance.py")
            return False
    else:
        print("Skipping evaluation (no threshold provided)")
        print(f"To evaluate: python DeepConvDTI/evaluate_performance.py {test_output} -n predict -T <threshold>")
    
    print("DeepConvDTI prediction completed")
    print(f"Model: {model_file}")
    print(f"Results: {test_output}")
    
    return True

def run_graphdta(args):
    """Run GraphDTA-based DTI prediction task (2 steps: create data, train/validate)"""
    print("Running GraphDTA-based DTI prediction...")

    embedding_file = args.embedding_file
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file not found: {embedding_file}")
        return False

    if not os.path.exists("GraphDTA"):
        print("Error: GraphDTA directory not found")
        return False

    required_files = ["GraphDTA/create_data.py", "GraphDTA/training_validation.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required GraphDTA files:")
        for f in missing_files:
            print(f"  - {f}")
        return False

    raw_data_dir = "GraphDTA/data/mydata"
    train_file = os.path.join(raw_data_dir, "train_merged.csv")
    test_file = os.path.join(raw_data_dir, "test_merged.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Error: Missing raw data files:")
        print(f"  Expected: {train_file}")
        print(f"  Expected: {test_file}")
        print("Please ensure the raw CSV files are present in GraphDTA/data/mydata/")
        return False

    original_dir = os.getcwd()
    try:
        os.chdir("GraphDTA")

        print("Creating processed data...")
        create_data_cmd = [
            "python", "create_data.py",
            "--embedding_file", os.path.join("..", embedding_file),
            "--embedding_dim", str(args.dim),
            "--raw_dir", "data/mydata",
            "--proc_dir", "data/processed",
            "--train_file", "train_merged.csv",
            "--test_file", "test_merged.csv",
            "--output_train_file", "mydata_train_lm.pt",
            "--output_test_file", "mydata_test_lm.pt"
        ]
        
        try:
            result = subprocess.run(create_data_cmd, check=True)
            print("Data creation completed")
        except subprocess.CalledProcessError as e:
            print(f"Error: Data creation failed with exit code {e.returncode}")
            return False
        
        print("Training model...")

        model_mapping = {
            "gin": 0
        }
        
        model_idx = model_mapping.get(args.graphdta_model.lower(), 0)
        
        train_cmd = [
            "python", "training_validation.py",
            str(model_idx),
            str(args.gpu_id),
            "--embedding_dim", str(args.dim),
            "--epochs", str(args.epochs),
            "--lr", str(args.learning_rate),
            "--train_batch_size", str(args.batch_size),
            "--val_batch_size", str(args.batch_size),
            "--test_batch_size", str(args.batch_size),
            "--data_dir", "data/processed",
            "--train_dataset_file", "mydata_train_lm.pt",
            "--test_dataset_file", "mydata_test_lm.pt",
            "--save_model_name", f"model_val_lm_{args.graphdta_model}",
            "--result_file_name", f"result_val_lm_{args.graphdta_model}",
            "--val_split_ratio", str(args.val_split_ratio),
            "--log_interval", str(args.log_interval),
            "--num_workers", str(args.num_workers)
        ]
        
        if args.no_lm:
            train_cmd.append("--no_lm")
        
        try:
            result = subprocess.run(train_cmd, check=True)
            print("Training completed")
        except subprocess.CalledProcessError as e:
            print(f"Error: Training failed with exit code {e.returncode}")
            return False
        
        print("GraphDTA prediction completed")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Run downstream tasks using LM embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        "task",
        choices=["mlp", "gnn", "deepconv", "graphdta"],
        help="Downstream task to run: 'mlp' for MLP-based DTI, 'gnn' for GNN-based DTI, 'deepconv' for DeepConvDTI, 'graphdta' for GraphDTA"
    )
    
    parser.add_argument(
        "--embedding-file",
        type=str,
        required=True,
        help="Path to embedding file (.pt). This parameter is required."
    )
    
    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Embedding dimension. This parameter is required."
    )
    
    parser.add_argument(
        "--lm-embedding-size",
        type=int,
        help="LLM embedding dimension for MLP (deprecated, use --dim instead)"
    )
    
    parser.add_argument(
        "--fc-layers",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Fully connected layer sizes for MLP/DeepConvDTI (default: 512 256 128)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate for MLP/DeepConvDTI (default: 0.0001)"
    )
    
    parser.add_argument(
        "--decay",
        type=float,
        default=0.0001,
        help="Learning rate decay for MLP/DeepConvDTI (default: 0.0001)"
    )
    
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        help="Activation function for MLP/DeepConvDTI (default: elu)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for MLP/DeepConvDTI (default: 32)"
    )
    
    parser.add_argument(
        "--save-emb-epochs",
        type=str,
        help="Specify epochs to save embeddings for MLP, separated by commas, e.g. '10,20,30'"
    )
    
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./",
        help="Output directory for saving embedding files (default: ./)"
    )

    parser.add_argument(
        "--gnn-model",
        type=str,
        choices=["lightgcn", "ngcf"],
        default="lightgcn",
        help="GNN model type: 'lightgcn' or 'ngcf' (default: lightgcn)"
    )

    parser.add_argument(
        "--drug-layers",
        type=int,
        nargs="+",
        default=[512, 128],
        help="Dense layers for drugs in DeepConvDTI (default: 512 128)"
    )
    
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 30],
        help="Window sizes for convolution in DeepConvDTI (default: 10 15 20 25 30)"
    )
    
    parser.add_argument(
        "--protein-layers",
        type=int,
        nargs="+",
        default=[128],
        help="Dense layers for proteins in DeepConvDTI (default: 128)"
    )
    
    parser.add_argument(
        "--prot-vec",
        type=str,
        default="Convolution",
        help="Type of protein feature for DeepConvDTI (default: Convolution)"
    )
    
    parser.add_argument(
        "--prot-len",
        type=int,
        default=2500,
        help="Protein vector length for DeepConvDTI (default: 2500)"
    )
    
    parser.add_argument(
        "--drug-vec",
        type=str,
        default="morgan_fp_r2",
        help="Type of drug feature for DeepConvDTI (default: morgan_fp_r2)"
    )
    
    parser.add_argument(
        "--drug-len",
        type=int,
        default=2048,
        help="Drug vector length for DeepConvDTI (default: 2048)"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout ratio for DeepConvDTI (default: 0.2)"
    )
    
    parser.add_argument(
        "--n-filters",
        type=int,
        default=128,
        help="Number of filters for convolution layer in DeepConvDTI (default: 128)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for evaluation in DeepConvDTI (e.g., 0.642221). If not provided, evaluation step will be skipped."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30 for MLP/DeepConvDTI, use 1000+ for GNN)"
    )

    parser.add_argument(
        "--graphdta-model",
        type=str,
        choices=["gin"],
        default="gin",
        help="GraphDTA model type: only 'gin' is available (default: gin)"
    )
    
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for training GraphDTA (default: 0)"
    )
    
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for GraphDTA (default: 0.2)"
    )
    
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval for GraphDTA training (default: 10)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for GraphDTA data loading (default: 4)"
    )
    
    parser.add_argument(
        "--no-lm",
        action="store_true",
        help="Flag to exclude LM embeddings in GraphDTA"
    )
    
    args = parser.parse_args()

    if args.task == "mlp":
        if args.lm_embedding_size is not None:
            print("Warning: --lm-embedding-size is deprecated, using --dim instead")
            if args.dim != args.lm_embedding_size:
                print(f"Note: Using --dim={args.dim} (ignoring --lm-embedding-size={args.lm_embedding_size})")
        args.lm_embedding_size = args.dim

    if args.task == "gnn" and args.epochs == 30:
        args.epochs = 1000
        print(f"Using default {args.epochs} epochs for GNN task")
    elif args.task == "deepconv" and args.epochs == 30:
        args.epochs = 20
        print(f"Using default {args.epochs} epochs for DeepConvDTI task")
    elif args.task == "graphdta" and args.epochs == 30:
        args.epochs = 100
        print(f"Using default {args.epochs} epochs for GraphDTA task")

    if args.task == "graphdta" and args.learning_rate == 0.0001:
        args.learning_rate = 5e-4
        print(f"Using default learning rate {args.learning_rate} for GraphDTA task")
    
    print("DrugLM Downstream Task Runner")
    print(f"Task: {args.task}")
    print(f"Embedding file: {args.embedding_file}")
    print(f"Embedding dimension: {args.dim}")
    if args.task == "gnn":
        print(f"GNN Model: {args.gnn_model}")
    elif args.task == "deepconv":
        if args.threshold:
            print(f"Evaluation threshold: {args.threshold}")
    elif args.task == "graphdta":
        print(f"Model: {args.graphdta_model}")
        print(f"Use LM embeddings: {not args.no_lm}")
    print(f"Epochs: {args.epochs}")
    
    if args.task == "mlp":
        success = run_mlp_dti(args)
    elif args.task == "gnn":
        success = run_gnn_dti(args)
    elif args.task == "deepconv":
        success = run_deepconv_dti(args)
    elif args.task == "graphdta":
        success = run_graphdta(args)
    else:
        print(f"Error: Unknown task '{args.task}'")
        sys.exit(1)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 