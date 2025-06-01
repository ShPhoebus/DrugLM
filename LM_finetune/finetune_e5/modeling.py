import logging
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Union, Dict, List

logger = logging.getLogger(__name__)

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

class MiniLMModel(torch.nn.Module):
    def __init__(
        self,
        base_model: AutoModel,
        tokenizer: AutoTokenizer = None,
        temperature: float = 0.05,
        normalize_embeddings: bool = True,
        frozen_layers: int = 22,
        freeze_pooler: bool = True
    ):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.frozen_layers_config = frozen_layers
        self.freeze_pooler_config = freeze_pooler

        self.use_e5_average_pooling = False
        if hasattr(self.model, 'name_or_path') and "e5" in self.model.name_or_path.lower():
             self.use_e5_average_pooling = True
             logger.info(f"Detected E5 model ({self.model.name_or_path}), using E5-style average pooling.")
        elif hasattr(self.model.config, 'name_or_path') and "e5" in self.model.config.name_or_path.lower():
            self.use_e5_average_pooling = True
            logger.info(f"Detected E5 model ({self.model.config.name_or_path}), using E5-style average pooling.")
        else:
            logger.info(f"Model ({getattr(self.model.config, 'name_or_path', 'unknown')}) is not E5, using CLS token pooling.")
        
        self._freeze_layers()

    def _freeze_layers(self):
        num_encoder_layers = 0
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            num_encoder_layers = len(self.model.encoder.layer)
        else:
            logger.warning("Model does not have a standard encoder structure. Layer freezing might be skipped or incomplete.")

        freeze_upto_idx = self.frozen_layers_config
        if num_encoder_layers > 0:
             freeze_upto_idx = min(freeze_upto_idx, num_encoder_layers -1)

        logger.info(f"Freezing encoder layers from 0 up to (but not including) index: {freeze_upto_idx} (out of {num_encoder_layers} layers). Layers from {freeze_upto_idx} onwards will be trainable.")

        if hasattr(self.model, 'embeddings') and self.model.embeddings is not None:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
        
        if num_encoder_layers > 0:
            for i in range(num_encoder_layers):
                layer = self.model.encoder.layer[i]
                should_freeze_this_layer = (i < freeze_upto_idx)
                for param in layer.parameters():
                    param.requires_grad = not should_freeze_this_layer
            
        if hasattr(self.model, 'pooler') and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = not self.freeze_pooler_config
            logger.info(f"Pooler layer exists. Freeze state: {self.freeze_pooler_config}")
        else:
            logger.info("Model does not have a pooler layer.")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        percentage_str = f"{(trainable_params/total_params):.1%}" if total_params > 0 else "0.0%"
        logger.info(f"Trainable params: {trainable_params}/{total_params} ({percentage_str})")

    def encode(self, features: Dict[str, Tensor]) -> Tensor:
        outputs = self.model(**features, return_dict=True)
        
        if self.use_e5_average_pooling:
            embeddings = average_pool(outputs.last_hidden_state, features['attention_mask'])
        else:
            embeddings = outputs.last_hidden_state[:, 0]
            
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def forward(self, 
                query_input_ids: Tensor,
                query_attention_mask: Tensor,
                positive_input_ids: Tensor,
                positive_attention_mask: Tensor,
                negative_input_ids: Tensor,
                negative_attention_mask: Tensor) -> Tensor:
        
        query_embeds = self.encode({
            'input_ids': query_input_ids,
            'attention_mask': query_attention_mask
        })
        
        positive_embeds = self.encode({
            'input_ids': positive_input_ids,
            'attention_mask': positive_attention_mask
        })
        
        negative_embeds = self.encode({
            'input_ids': negative_input_ids,
            'attention_mask': negative_attention_mask
        })

        batch_size = query_embeds.size(0)
        num_negatives_per_query = negative_embeds.size(0) // batch_size
        
        negative_embeds_reshaped = negative_embeds.view(batch_size, num_negatives_per_query, -1)

        positive_scores = torch.sum(query_embeds * positive_embeds, dim=1) / self.temperature
        
        negative_scores = torch.bmm(query_embeds.unsqueeze(1), 
                                  negative_embeds_reshaped.transpose(1, 2)).squeeze(1)
        negative_scores = negative_scores / self.temperature

        scores = torch.cat([
            positive_scores.unsqueeze(-1),
            negative_scores
        ], dim=-1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
        return self.cross_entropy(scores, labels)

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
