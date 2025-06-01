def main():
    config = {
        "model_name_or_path": "intfloat/e5-large-v2",
        "cache_dir": "./cache/model",
        "trust_remote_code": True,
        "train_data": "LM_finetune/training_data_noSMILES2SeqInChI.jsonl",
        "cache_path": "./cache/data",
        "train_group_size": 8,
        "max_length": 512,
        "pad_to_multiple_of": 8,
        "output_dir": "./e5_output",
        "overwrite_output_dir": True,
        "learning_rate": 5e-7,
        "num_train_epochs": 7,
        "per_device_train_batch_size": 1,
        "dataloader_drop_last": True,
        "warmup_steps": 3000,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "temperature": 0.05,
        "normalize_embeddings": True,
        "gradient_accumulation_steps": 8,
        "fp16": True,
        "embedding_dim": 1024,
        "weight_decay": 0.1,    
        "frozen_layers": 22,  
        "use_progressive_unfreeze": False, 
        "freeze_pooler": True,
    }
    
    from .runner import MiniLMRunner
    runner = MiniLMRunner(config=config)
    runner.run()

if __name__ == "__main__":
    main()
