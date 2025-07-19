import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

with open('LM_finetune/text_lists_noSMILES2SeqInChI.json', 'r', encoding='utf-8') as f:
    text_lists = json.load(f)

with open('LM_finetune/id_mappings.json', 'r', encoding='utf-8') as f:
    id_mappings = json.load(f)

drug_ids = list(id_mappings['drug'].keys())
target_ids = list(id_mappings['target'].keys())

model = SentenceTransformer('intfloat/e5-large-v2')
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

drug_embeddings = []
drug_texts = text_lists['drug_texts']
print("Generating drug embedding...")
for drug_id in tqdm(drug_ids):
    text = drug_texts[int(drug_id)]
    text = "query: " + text
    embedding = model.encode(text, convert_to_tensor=True)
    drug_embeddings.append(embedding)

drug_embeddings = torch.stack(drug_embeddings)

target_embeddings = []
target_texts = text_lists['target_texts']
print("Generating target embedding...")
for target_id in tqdm(target_ids):
    text = target_texts[int(target_id)]
    text = "passage: " + text
    embedding = model.encode(text, convert_to_tensor=True)
    target_embeddings.append(embedding)

target_embeddings = torch.stack(target_embeddings)

print("Saving embeddings...")
torch.save({
    'drug_embeddings': drug_embeddings,
    'target_embeddings': target_embeddings,
    'drug_ids': drug_ids,
    'target_ids': target_ids
}, 'e5_pretrained_embedding.pt')

print("Saved embeddings")
