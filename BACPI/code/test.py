import torch

data = torch.load('../data/interaction/our_data/mpnet_noSMILES2SeqInChI_embeddings.pt')
drug_embeddings = data['drug_embeddings']
target_embeddings = data['target_embeddings']
drug_ids = data['drug_ids']
target_ids = data['target_ids']

drug_embedding_list = drug_embeddings.tolist()
drug_dict = dict(zip(drug_ids, drug_embedding_list))
print(len(drug_embeddings[0]))

target_embedding_list = target_embeddings.tolist()
target_dict = dict(zip(target_ids, target_embedding_list))
print(len(target_embeddings[0]))

print(target_ids)

# print(target_dict.keys())