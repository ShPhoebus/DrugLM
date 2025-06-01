import logging
from typing import Tuple
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    TrainingArguments
)
from torch.utils.data import Dataset
import json
import torch

from .modeling import MiniLMModel
from .trainer import MiniLMTrainer

logger = logging.getLogger(__name__)

class MiniLMDataCollator:
    
    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: int = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        batch = {
            'query_input_ids': self._pad_sequence([f['query_input_ids'] for f in features]),
            'query_attention_mask': self._pad_sequence([f['query_attention_mask'] for f in features]),
            'positive_input_ids': self._pad_sequence([f['positive_input_ids'] for f in features]),
            'positive_attention_mask': self._pad_sequence([f['positive_attention_mask'] for f in features]),
        }

        all_negative_input_ids = []
        all_negative_attention_mask = []
        for f in features:
            for neg_ids, neg_mask in zip(f['negative_input_ids'], f['negative_attention_mask']):
                all_negative_input_ids.append(neg_ids)
                all_negative_attention_mask.append(neg_mask)
        
        batch['negative_input_ids'] = self._pad_sequence(all_negative_input_ids)
        batch['negative_attention_mask'] = self._pad_sequence(all_negative_attention_mask)
        
        return batch
    
    def _pad_sequence(self, sequences):
        padded = self.tokenizer.pad(
            {'input_ids': sequences},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )['input_ids']
        return padded

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)
        logger.info(f"Loaded {len(self.data)} training samples")
        
    def load_data(self, file_path: str):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    'query': item['query'],
                    'positive': item['pos'][0],
                    'negatives': item['neg']
                })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        query_encoding = self.tokenizer(
            item['query'],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        positive_encoding = self.tokenizer(
            item['positive'],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        negative_encodings = self.tokenizer(
            item['negatives'],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'],
            'query_attention_mask': query_encoding['attention_mask'],
            'positive_input_ids': positive_encoding['input_ids'],
            'positive_attention_mask': positive_encoding['attention_mask'],
            'negative_input_ids': negative_encodings['input_ids'],
            'negative_attention_mask': negative_encodings['attention_mask']
        }

class MiniLMRunner:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        tokenizer, model = self._load_tokenizer_and_model()
        
        train_dataset = TextDataset(
            self.config["train_data"],
            tokenizer,
            self.config["max_length"]
        )
        
        data_collator = MiniLMDataCollator(
            tokenizer,
            pad_to_multiple_of=self.config["pad_to_multiple_of"]
        )
        
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            learning_rate=self.config["learning_rate"],
            warmup_steps=self.config["warmup_steps"],
            logging_steps=self.config["logging_steps"],
            save_strategy=self.config["save_strategy"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            fp16=self.config["fp16"],
            dataloader_drop_last=self.config["dataloader_drop_last"],
            overwrite_output_dir=self.config["overwrite_output_dir"]
        )
        
        trainer = MiniLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            max_length=self.config["max_length"]
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        trainer.save_model()
        logger.info(f"Model saved to {self.config['output_dir']}")
    
    def _load_tokenizer_and_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name_or_path"],
            cache_dir=self.config["cache_dir"],
            trust_remote_code=self.config["trust_remote_code"]
        )
        
        base_model = AutoModel.from_pretrained(
            self.config["model_name_or_path"],
            cache_dir=self.config["cache_dir"],
            trust_remote_code=self.config["trust_remote_code"],
            add_pooling_layer=True
        )
        
        model = MiniLMModel(
            base_model=base_model,
            tokenizer=tokenizer,
            temperature=self.config["temperature"],
            normalize_embeddings=self.config["normalize_embeddings"]
        )
        
        return tokenizer, model

    def _create_optimizer(self, model):
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameters:{name}")
                trainable_params.append(param)
        
        logger.info(f"Number of trainable parameters: {len(trainable_params)}")
        logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
