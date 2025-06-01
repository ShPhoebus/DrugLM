import logging
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Union, Dict, List

logger = logging.getLogger(__name__)

class MiniLMModel(torch.nn.Module):
    def __init__(
        self,
        base_model: AutoModel,
        tokenizer: AutoTokenizer = None,
        temperature: float = 0.05,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        
        assert hasattr(self.model, 'encoder'), "Model must contain encoder module"
        assert len(self.model.encoder.layer) >= 24, "Model must have at least 24 layers"
        
        self._freeze_layers()

    def _freeze_layers(self):
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
            
        for layer_num in range(22):
            layer = self.model.encoder.layer[layer_num]
            for param in layer.parameters():
                param.requires_grad = False
        
        for layer_num in [22, 23]:
            layer = self.model.encoder.layer[layer_num]
            for param in layer.parameters():
                param.requires_grad = True
                
        if hasattr(self.model, 'pooler') and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = True
        else:
            print("Warning: Model has no pooler layer, skipping pooler parameter unfreezing")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameter ratio: {trainable_params/total_params:.1%}")

    def encode(self, features: Union[List, Dict]) -> torch.Tensor:
        if features is None:
            return None

        if not isinstance(features, list):
            last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
            embeddings = last_hidden_state[:, 0]
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            return embeddings.contiguous()

        all_embeddings = []
        for sub_features in features:
            last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
            embeddings = last_hidden_state[:, 0] 
            all_embeddings.append(embeddings)
        
        all_embeddings = torch.cat(all_embeddings, 0).contiguous()
        if self.normalize_embeddings:
            all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1)
        return all_embeddings.contiguous()

    def forward(self, 
                query_input_ids: torch.Tensor,
                query_attention_mask: torch.Tensor,
                positive_input_ids: torch.Tensor,
                positive_attention_mask: torch.Tensor,
                negative_input_ids: torch.Tensor,
                negative_attention_mask: torch.Tensor) -> torch.Tensor:

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
        neg_per_query = negative_embeds.size(0) // batch_size
        negative_embeds = negative_embeds.view(batch_size, neg_per_query, -1)

        positive_scores = torch.sum(query_embeds * positive_embeds, dim=1) / self.temperature
        negative_scores = torch.bmm(
            query_embeds.unsqueeze(1),
            negative_embeds.transpose(-2, -1)
        ).squeeze(1) / self.temperature

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
