import os
import torch
import logging
import math
import json
from typing import Optional, Dict, Union
from torch import Tensor
from transformers import Trainer, TrainingArguments
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

class MiniLMTrainer(Trainer):
    def __init__(self, 
                 model=None,
                 args: TrainingArguments = None,
                 train_dataset=None,
                 data_collator=None,
                 tokenizer=None,
                 max_length=None,
                 config=None):

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
        
        self.config = config
    
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
        
        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2 
            for p in model.parameters() 
            if p.requires_grad and p.grad is not None
        ) ** 0.5

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                logger.debug(f"Gradient monitoring: {name} | mean={param.grad.mean():.2e} norm={param.grad.norm():.2e}")
        
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
        if not self.text_lists or not self.id_mappings:
            logger.error("Cannot generate embeddings: text_lists or id_mappings not loaded.")
            return

        logger.info(f"E5-Style Embedding Generation: Starting for checkpoint {checkpoint_path}")
        
        model_hf_identifier = self.config.get("model_name_or_path", "intfloat/e5-large-v2")
        if "e5" not in model_hf_identifier.lower():
             logger.warning(f"E5-style embedding generation called, but model is {model_hf_identifier}")

        base_model_name_for_filename = model_hf_identifier.split('/')[-1]
        is_best = "best" in checkpoint_path
        epoch_str = checkpoint_path.split("epoch-")[-1].split("-")[0] if "epoch-" in checkpoint_path else "final"
        
        emb_model = AutoModel.from_pretrained(model_hf_identifier, trust_remote_code=True)
        emb_tokenizer = AutoTokenizer.from_pretrained(model_hf_identifier, trust_remote_code=True)
        
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            logger.error(f"E5-Style Embedding Generation: Weights file not found at {weights_path}")
            return
        
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        load_info = emb_model.load_state_dict(state_dict, strict=False)
        logger.info(f"E5-Style Embedding Generation: Loaded weights into emb_model. Missing: {load_info.missing_keys}, Unexpected: {load_info.unexpected_keys}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb_model = emb_model.to(device)
        emb_model.eval()
                    
        def generate_single_e5_embedding(text_content):
            prefixed_text = f"query: {text_content}"
            inputs = emb_tokenizer(
                prefixed_text,
                padding=True, truncation=True,
                max_length=self.config.get("max_length", 512),
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = emb_model(**inputs)
                pooled_embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
            return normalized_embeddings.cpu()

        drug_embeddings_list = []
        for drug_id in tqdm(self.drug_ids, desc=f"Drug Embeddings (E5) for {os.path.basename(checkpoint_path)}"):
            text = self.text_lists['drug_texts'][int(drug_id)]
            drug_embeddings_list.append(generate_single_e5_embedding(text))
        
        target_embeddings_list = []
        for target_id in tqdm(self.target_ids, desc=f"Target Embeddings (E5) for {os.path.basename(checkpoint_path)}"):
            text = self.text_lists['target_texts'][int(target_id)]
            target_embeddings_list.append(generate_single_e5_embedding(text))

        if not drug_embeddings_list or not target_embeddings_list:
            logger.error(f"E5-Style Embedding Generation: Failed to generate one or both sets of embeddings for {checkpoint_path}.")
            return
            
        final_drug_embeddings = torch.cat(drug_embeddings_list, dim=0)
        final_target_embeddings = torch.cat(target_embeddings_list, dim=0)

        output_filename = f"{base_model_name_for_filename}_epoch{epoch_str}_{'best' if is_best else 'normal'}_e5avgpool_embeddings.pt"
        output_path = os.path.join(checkpoint_path, output_filename)
        
        torch.save({
            'drug_embeddings': final_drug_embeddings,
            'target_embeddings': final_target_embeddings,
            'drug_ids': self.drug_ids,
            'target_ids': self.target_ids,
            'embedding_type': 'E5-AveragePooling'
        }, output_path)
        logger.info(f"E5-Style Embeddings saved to {output_path} (Drugs: {final_drug_embeddings.shape}, Targets: {final_target_embeddings.shape})")

    def _init_model(self):
        total_layers = self.model.config.num_hidden_layers
        freeze_num = min(self.args.frozen_layers, total_layers)
        
        for layer in self.model.encoder.layer[:freeze_num]:
            for param in layer.parameters():
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameter count: {trainable_params/1e6:.2f}M")
