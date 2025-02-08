import json
import ipdb
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

def get_bert_embedding(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def embeddings_to_list(embeddings):
    return embeddings.cpu().tolist()

if __name__ == '__main__':

    device = torch.device("cuda:1")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device)

    results = []
    with open('file_name_1.pkl', 'rb') as f1:
        database = pickle.load(f1)

    
    for case_name, case_data in tqdm(database.items(), desc="Processing cases"):

        Q_cot = case_data[0]
        Q_cot_join = ", ".join(Q_cot)
        

        embedding = get_bert_embedding(Q_cot_join)

        embedding_list = embeddings_to_list(embedding)

        result = {
            "case_name": case_name,
            "embedding": embedding_list
        }
        results.append(result)


    with open('file_name_2.json', 'w') as f:
        json.dump(results, f)
