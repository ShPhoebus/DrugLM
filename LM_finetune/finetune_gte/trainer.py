import os
import torch
import logging
import math
import json
from typing import Optional, Dict, Union
from transformers import Trainer, TrainingArguments
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

class MiniLMTrainer(Trainer):
    def __init__(self, 
                 model=None,
                 args: TrainingArguments = None,
                 train_dataset=None,
                 data_collator=None,
                 tokenizer=None,
                 max_length=None):

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        print("="*50)
        print("MiniLMTrainer initialization...")

        self.callback_handler.add_callback(self)
        
        try:
            print(f"Training set size: {len(self.train_dataset)}")
            print(f"Batch size: {self.args.per_device_train_batch_size}")
            print(f"Total epochs: {self.args.num_train_epochs}")
            print(f"Steps per epoch: {len(self.get_train_dataloader())}")
            print(f"Total training steps: {len(self.get_train_dataloader()) * self.args.num_train_epochs}")
        except Exception as e:
            print(f"Error getting training info: {e}")
        print("="*50)
        
        self._epoch = 0
        self.best_loss = float('inf')
        self.best_step = 0
        self.max_length = max_length
        self.current_epoch_losses = []
        self.epoch_losses = []
        self.current_epoch_best_loss = float('inf')
        self.current_epoch_best_state = None
    
        try:
            self._total_steps_per_epoch = len(self.get_train_dataloader())
        except Exception as e:
            print(f"Cannot calculate total_steps_per_epoch during initialization: {e}")
            self._total_steps_per_epoch = None
        
        with open('LM_finetune/text_lists_noSMILES2SeqInChI.json', 'r', encoding='utf-8') as f:
            self.text_lists = json.load(f)
        with open('LM_finetune/id_mappings.json', 'r', encoding='utf-8') as f:
            self.id_mappings = json.load(f)

        self.drug_ids = list(self.id_mappings['drug'].keys())
        self.target_ids = list(self.id_mappings['target'].keys())
    
    def generate_embedding(self, text):
        device = next(self.model.parameters()).device
        
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def generate_and_save_embeddings(self, epoch, is_best=False):
        logger.info(f"Generating embeddings for {'best model' if is_best else f'epoch {epoch}'}...")
        self.model.eval()
        
        device = next(self.model.parameters()).device
        
        drug_embeddings = []
        drug_texts = self.text_lists['drug_texts']
        for drug_id in tqdm(self.drug_ids, desc="Processing drugs"):
            text = drug_texts[int(drug_id)]
            embedding = self.generate_embedding(text)
            drug_embeddings.append(embedding.cpu())
        drug_embeddings = torch.cat(drug_embeddings, dim=0)
        
        target_embeddings = []
        target_texts = self.text_lists['target_texts']
        for target_id in tqdm(self.target_ids, desc="Processing targets"):
            text = target_texts[int(target_id)]
            embedding = self.generate_embedding(text)
            target_embeddings.append(embedding.cpu())
        target_embeddings = torch.cat(target_embeddings, dim=0)
        
        if is_best:
            output_dir = os.path.join(self.args.output_dir, "best_model")
        else:
            output_dir = os.path.join(self.args.output_dir, f"epoch-{epoch}")
        
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'drug_embeddings': drug_embeddings,
            'target_embeddings': target_embeddings,
            'drug_ids': self.drug_ids,
            'target_ids': self.target_ids
        }, os.path.join(output_dir, 'embeddings.pt'))
        
        logger.info(f"Embeddings saved to {output_dir}")
        logger.info(f"Drug embeddings shape: {drug_embeddings.shape}")
        logger.info(f"Target embeddings shape: {target_embeddings.shape}")
        
        self.model.train()
    
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        batch_loss = loss.item()
        
        self.current_epoch_losses.append(batch_loss)
        
        if batch_loss < self.current_epoch_best_loss:
            logger.info(f"Found better batch loss: {batch_loss:.4f} (previous: {self.current_epoch_best_loss:.4f})")
            self.current_epoch_best_loss = batch_loss
            self.current_epoch_best_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
        
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        
        if return_outputs:
            return loss, None
        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Executing model save to {output_dir}")
        
        try:
            if self.is_world_process_zero():
                new_state_dict = {
                    k.replace('model.', ''): v 
                    for k, v in self.model.state_dict().items()
                }
                model_bin_path = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(new_state_dict, model_bin_path)
            
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                
                info = {
                    'epoch': self._epoch,
                    'global_step': self.state.global_step,
                    'loss': self.state.log_history[-1].get('loss', 'unknown') if self.state.log_history else 'unknown',
                    'best_loss': self.current_epoch_best_loss if hasattr(self, 'current_epoch_best_loss') else None,
                }
                info_path = os.path.join(output_dir, "training_info.pt")
                torch.save(info, info_path)
                
                print(f"Model weights saved to {model_bin_path}")
                
        except Exception as e:
            print(f"Error saving model: {e}")

    def on_train_begin(self, args, state, control, **kwargs):
        frozen_params = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in self.model.parameters())
        logger.info(f"Frozen parameter ratio: {frozen_params/total_params:.1%}")
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = math.ceil(state.epoch) + 1
        print(f"Starting training for {current_epoch}th epoch...")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if len(self.current_epoch_losses) == 0:
            return control
        current_epoch = math.floor(state.epoch) + 1
        
        if current_epoch > self._epoch:
            epoch_output_dir = os.path.join(self.args.output_dir, f"epoch-{current_epoch}")
            print(f"Starting to save {current_epoch}th epoch model...")
            self.save_model(epoch_output_dir)
            print(f"Completed saving {current_epoch}th epoch model to {epoch_output_dir}")
            
            if self.current_epoch_best_state is not None:
                current_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                try:
                    self.model.load_state_dict(self.current_epoch_best_state)
                    epoch_best_dir = f"{self.args.output_dir}/epoch-{current_epoch}-best"
                    os.makedirs(epoch_best_dir, exist_ok=True)
                    self.save_model(epoch_best_dir)
                    print(f"Saved best model (loss: {self.current_epoch_best_loss:.4f}) to {epoch_best_dir}")
                    
                finally:
                    self.model.load_state_dict(current_state)
                    del current_state
            
            avg_epoch_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            self.epoch_losses.append(avg_epoch_loss)
            
            print(f"Average loss for epoch {current_epoch}: {avg_epoch_loss:.4f}")
            
            try:
                self._generate_embeddings_for_checkpoint(epoch_output_dir)
                
                if self.current_epoch_best_state is not None:
                    epoch_best_dir = os.path.join(self.args.output_dir, f"epoch-{current_epoch}-best")
                    os.makedirs(epoch_best_dir, exist_ok=True)
                    
                    torch.save(
                        self.current_epoch_best_state, 
                        os.path.join(epoch_best_dir, "pytorch_model.bin")
                    )
                    self._generate_embeddings_for_checkpoint(epoch_best_dir)
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
            
            self.current_epoch_losses = []
            self.current_epoch_best_loss = float('inf')
            self.current_epoch_best_state = None
            self._epoch = current_epoch
        
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        return control

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training completed...")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        return control

    def on_save(self, args, state, control, **kwargs):
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control

    def _generate_embeddings_for_checkpoint(self, checkpoint_path: str):
        logger.info(f"Starting to generate embeddings for checkpoint {checkpoint_path}...")
        
        base_model_name = "gte-large-en-v1.5"
        is_best = "best" in checkpoint_path
        epoch = checkpoint_path.split("epoch-")[-1].split("-")[0] if "epoch" in checkpoint_path else "final"
        
        model = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-large-en-v1.5",
            trust_remote_code=True,
            add_pooling_layer=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Alibaba-NLP/gte-large-en-v1.5", 
            trust_remote_code=True
        )
        
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            logger.error(f"Weight file not found: {weights_path}")
            return
        
        state_dict = torch.load(weights_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        with open('LM_finetune/text_lists_noSMILES2SeqInChI.json', 'r', encoding='utf-8') as f:
            text_lists = json.load(f)
        with open('LM_finetune/id_mappings.json', 'r', encoding='utf-8') as f:
            id_mappings = json.load(f)
        
        def generate_embeddings(text):
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0]
                pooler_embeddings = outputs.pooler_output
                
                cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                pooler_embeddings = torch.nn.functional.normalize(pooler_embeddings, p=2, dim=1)
            
            return {
                'cls': cls_embeddings.cpu(),
                'pooler': pooler_embeddings.cpu()
            }
            
        drug_embeddings_cls = []
        drug_embeddings_pooler = []
        drug_texts = text_lists['drug_texts']
        for drug_id in tqdm(list(id_mappings['drug'].keys()), desc="Processing drugs"):
            text = drug_texts[int(drug_id)]
            embeddings = generate_embeddings(text)
            drug_embeddings_cls.append(embeddings['cls'])
            drug_embeddings_pooler.append(embeddings['pooler'])
        
        target_embeddings_cls = []
        target_embeddings_pooler = []
        target_texts = text_lists['target_texts']
        for target_id in tqdm(list(id_mappings['target'].keys()), desc="Processing targets"):
            text = target_texts[int(target_id)]
            embeddings = generate_embeddings(text)
            target_embeddings_cls.append(embeddings['cls'])
            target_embeddings_pooler.append(embeddings['pooler'])
        
        name_template = f"{base_model_name}_epoch{epoch}_{'best' if is_best else 'normal'}"
        
        for embed_type in ['cls', 'pooler']:
            output_path = os.path.join(
                checkpoint_path, 
                f"{name_template}_{embed_type}_embeddings.pt"
            )
            
            torch.save({
                'drug_embeddings': torch.cat(eval(f'drug_embeddings_{embed_type}'), dim=0),
                'target_embeddings': torch.cat(eval(f'target_embeddings_{embed_type}'), dim=0),
                'drug_ids': list(id_mappings['drug'].keys()),
                'target_ids': list(id_mappings['target'].keys()),
                'embedding_type': embed_type.upper()
            }, output_path)
            logger.info(f"{embed_type.upper()} embeddings saved to {output_path}")
