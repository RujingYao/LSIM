import json
import random
from vllm import LLM, SamplingParams
import pandas as pd
import re
import torch.distributed as dist
import pickle
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import evaluate
import torch.optim.lr_scheduler as lr_scheduler
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertModel
import heapq
import re
from torch.nn import DataParallel
import os
from itertools import chain
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import ipdb
import numpy as np
from torch.distributions import Categorical
from transformers import BertTokenizer, BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from rouge import Rouge
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu

def get_split_dict(dict_train):
    names_train = []
    contents_train = []
    for name, content in dict_train.items():
        names_train.append(name) 
        contents_train.append(content)
    return names_train, contents_train


def get_embeddings(cases, model, embeddings_by_name):
    embeddings =  [model(embeddings_by_name[case].to(device)) for case in cases]
    return [embedding.squeeze() for embedding in embeddings]


from torch.utils.data import Dataset, DataLoader

class LegalDataset(Dataset):
    def __init__(self, data, train_embeddings_by_name, database_embeddings_by_name):
        self.data = data
        self.embeddings_by_name_train = train_embeddings_by_name
        self.embeddings_by_name_database = database_embeddings_by_name
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = list(self.data.keys())[idx] 
        item = self.data[key]           
        q_embedding = self.embeddings_by_name_train[key].to(device)

        similar_case_embeddings = [self.embeddings_by_name_database[name] for name in item['similar_case']]

        pos_scores = item['Qscore_posneg'][0]
        neg_scores = item['Qscore_posneg'][1]
        min_scores = item['Qscore_posneg'][2]
        
        pos_similar_case_embeddings_tensor = similar_case_embeddings[pos_scores[0]].to(device)
        neg_similar_case_embeddings_tensor = similar_case_embeddings[min_scores[0]].to(device)

        pos_state = torch.cat((q_embedding, pos_similar_case_embeddings_tensor), dim=1)
        neg_state = torch.cat((q_embedding, neg_similar_case_embeddings_tensor), dim=1)

        
        socres_int = [int(x) for x in item['Qscore']]
        socres_int_array = np.array(socres_int)


        return key, pos_state, neg_state, socres_int_array



class TestLegalDataset(Dataset):
    def __init__(self, data, train_embeddings_by_name, database_embeddings_by_name):
        self.data = data
        self.embeddings_by_name_train = train_embeddings_by_name
        self.embeddings_by_name_database = database_embeddings_by_name
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]  
        item = self.data[key]           
        q_embedding = self.embeddings_by_name_train[key].to(device)

        similar_case_embeddings = [self.embeddings_by_name_database[name] for name in item['similar_case']]
        similar_case_embeddings_tensor = torch.stack(similar_case_embeddings).to(device)
        state = [torch.cat((q_embedding, i), dim=1) for i in similar_case_embeddings_tensor]
        state = torch.stack(state).squeeze(1).to(device)
        
        
        scores = item['Qscore']
        socres_int = [int(x) for x in scores]
        socres_int_array = np.array(socres_int)
        return key, state, socres_int_array

class dssm(nn.Module):
    def __init__(self):
        super(dssm, self).__init__()
        self.l1 = nn.Linear(768*2, 600)
        nn.init.xavier_uniform_(self.l1.weight)
        self.l2 = nn.Linear(600, 300)
        nn.init.xavier_uniform_(self.l2.weight)
        self.l3 = nn.Linear(300, 128)
        nn.init.xavier_uniform_(self.l3.weight)
        self.out = nn.Linear(128, 1)  # 假设有30个不同的案例
        nn.init.xavier_uniform_(self.out.weight)
   
        
        
    def forward(self, x):

        x1 = F.tanh(self.l1(x))
        x2 = F.tanh(self.l2(x1))
        x3 = F.tanh(self.l3(x2))
        x_out = self.out(x3).squeeze(-1)
        return x_out



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    rouge = Rouge()
    meteor = evaluate.load('meteor')    
    

    with open('database_file_name.pkl', 'rb') as f1:
        database_orig = pickle.load(f1)
    with open('train_file_name.pkl', 'rb') as f2:
        train_orig = pickle.load(f2)
    with open('test_file_name.pkl', 'rb') as f3:
        test_orig = pickle.load(f3)   

    with open('emb_name_1.json', 'r') as f8:
        database_embeddings_dict = json.load(f8)
    database_embeddings_by_name = {item1['case_name']: torch.tensor(item1['embedding']) for item1 in database_embeddings_dict}

    
    with open('emb_name_2.json', 'r') as f9:
        train_embeddings_dict = json.load(f9)
    train_embeddings_by_name = {item2['case_name']: torch.tensor(item2['embedding']) for item2 in train_embeddings_dict}

    
    with open('emb_name_3.json', 'r') as f10:
        test_embeddings_dict = json.load(f10)
    test_embeddings_by_name = {item3['case_name']: torch.tensor(item3['embedding']) for item3 in test_embeddings_dict}    
     



    with open('cot_emb_name_1.json', 'r') as f8:
        database_cot_embeddings_dict = json.load(f8)
    database_cot_embeddings_by_name = {item1['case_name']: torch.tensor(item1['embedding']) for item1 in database_cot_embeddings_dict}

    
    with open('cot_emb_name_2.json', 'r') as f9:
        train_cot_embeddings_dict = json.load(f9)
    train_cot_embeddings_by_name = {item2['case_name']: torch.tensor(item2['embedding']) for item2 in train_cot_embeddings_dict}

    with open('cot_emb_name_3.json', 'r') as f10:
        test_cot_embeddings_dict = json.load(f10)
    test_cot_embeddings_by_name = {item3['case_name']: torch.tensor(item3['embedding']) for item3 in test_cot_embeddings_dict}    



    with open('score_file_1.pkl', 'rb') as f11:
        dict_single_scores_train = pickle.load(f11)  
    with open('score_file_2.pkl', 'rb') as f22:
        dict_single_scores_test = pickle.load(f22)  


    train = {}
    for key_3, value_3 in train_orig.items():
        train[key_3] = value_3
        train[key_3]["Qscore"] = dict_single_scores_train[key_3]    
    
    test = {}
    for key_3, value_3 in train_orig.items():
        test[key_3] = value_3
        test[key_3]["Qscore"] = dict_single_scores_test[key_3]     
  
    
    names_train_beforeshuffer, contents_train_beforeshuffer = get_split_dict(train) 
    names_database, contents_database = get_split_dict(database_orig)
    names_test, contents_test = get_split_dict(test) 
    print("len train: "+str(len(train)))
    print("len test: "+str(len(test)))
    random.seed(42)
        
    indexes = list(range(len(names_train_beforeshuffer)))
    random.shuffle(indexes)
    names_train = [names_train_beforeshuffer[i] for i in indexes]
    contents_train = [contents_train_beforeshuffer[j] for j in indexes]
 

    epochs =  50


    dssm_model = dssm().to(device)

    optimizer = torch.optim.Adam(dssm_model.parameters(), lr=0.0001, weight_decay=1e-3)

    action_save_all_epoch = []
    max_result = 0
    max_result_distance = 0
    max_result_epoch = 0
    max_result_epoch_distance = 0
    

    train_new = {}

    for one_name in names_train:
        all_scores_onesample = train[one_name]["Qscore"]
        max_socre = max(all_scores_onesample)
        min_score = min(all_scores_onesample)
        if int(max_socre) >2:
            positve_indices = [index for index, score in enumerate(all_scores_onesample) if score == max_socre]
            negative_indices = [index for index, score in enumerate(all_scores_onesample) if score != max_socre]
            min_indices = [index for index, score in enumerate(all_scores_onesample) if score == min_score]
            train_new[one_name] = train[one_name]
            train_new[one_name]["Qscore_posneg"] = [positve_indices, negative_indices, min_indices]
        

    train_dataset = LegalDataset(train_new, train_embeddings_by_name, database_embeddings_by_name)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)    
    test_dataset = TestLegalDataset(test, test_embeddings_by_name, database_embeddings_by_name)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  
    
    criterion = nn.MarginRankingLoss(margin=1)
    max_score_value = 0
    max_epoch = 0
    
    for epoch in range(epochs):
        dict_epoch_choosename = {}
        result_final = []
        epoch_loss = 0
        dssm_model.train()
        for batch_keys, pos_states, neg_states, batch_scores in train_dataloader:

            pos_pre_score = dssm_model(pos_states)
            neg_pre_score = dssm_model(neg_states)
            
   
            loss = criterion(pos_pre_score, neg_pre_score, torch.ones((neg_pre_score.shape[0], 1), device=device))
            

            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
            epoch_loss += loss.item()
            
        average_loss = epoch_loss / len(train_dataloader)

        print(f"Epoch {epoch}/{epochs} is completed, Epoch Loss:{average_loss}")
        

        dssm_model.eval()

        
        with torch.no_grad():
            max_scores_test = []
            for batch_keys_test, comb_states_test, batch_scores in test_dataloader:

                score_test = dssm_model(comb_states_test)
                top_values, top_indices = torch.topk(score_test, k=3)
                
                max_idx =  torch.max(score_test, dim=1).indices.to('cpu').tolist()

                for id_combine in range(len(top_values)):
                    
                    name_com = batch_keys_test[id_combine]
                    bm_25_30 = test[name_com]["similar_case"]
                    one_top_id = top_indices[id_combine]
                    pre_top_names = [bm_25_30[idx] for idx in one_top_id]
                    if name_com not in dict_epoch_choosename.keys():
                        dict_epoch_choosename[name_com] = pre_top_names
                
                for i, one_idx in enumerate(max_idx):
                    max_scores_test.append(batch_scores[i][one_idx].item())
            
            mean_score_test =  sum(max_scores_test)/len(max_scores_test)  

        if mean_score_test > max_score_value:
            max_score_value = mean_score_test
            max_epoch = epoch                

        save_path_name = "result.pkl"
        with open(save_path_name, 'wb') as f11:
            pickle.dump(dict_epoch_choosename, f11)  