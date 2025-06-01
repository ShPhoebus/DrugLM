# DrugLM

## Available Models

- **BGE**: BAAI/bge-large-en-v1.5
- **E5**: intfloat/e5-large-v2  
- **GTE**: Alibaba-NLP/gte-large-en-v1.5

## Usage

### Run Pretrained Models

```bash
# Run BGE pretrained model
python run_lm_model.py bge

# Run E5 pretrained model
python run_lm_model.py e5

# Run GTE pretrained model
python run_lm_model.py gte
```

### Run Fine-tuned Models

```bash
# Run BGE fine-tuned model
python run_lm_model.py bge --finetune

# Run E5 fine-tuned model
python run_lm_model.py e5 --finetune

# Run GTE fine-tuned model
python run_lm_model.py gte --finetune
```

### Run Downstream Tasks

```bash
# MLP-based DTI prediction
python run_downstream_task.py mlp --embedding-file bge_NonFT.pt --dim 1024

# GNN-based DTI prediction with LightGCN
python run_downstream_task.py gnn --embedding-file bge_NonFT.pt --dim 1024 --gnn-model lightgcn --epochs 1000

# GNN-based DTI prediction with NGCF
python run_downstream_task.py gnn --embedding-file bge_NonFT.pt --dim 1024 --gnn-model ngcf --epochs 1000

# DeepConvDTI-based DTI prediction
python run_downstream_task.py deepconv --embedding-file bge_NonFT.pt --dim 1024 --epochs 30

# GraphDTA-based DTI prediction
python run_downstream_task.py graphdta --embedding-file bge_NonFT.pt --dim 1024 --epochs 100
```

## Requirements

```bash
pip install -r requirements.txt
```

``` 