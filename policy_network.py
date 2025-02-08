import json
import random
from vllm import LLM, SamplingParams
import pandas as pd
import re
import torch.distributed as dist
import pickle
import warnings
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertModel
import heapq
import re
from torch.nn import DataParallel
import os
from itertools import chain
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.distributions import Categorical
from transformers import BertTokenizer, BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import networkx as nx

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



   
class PolicyNetwork(nn.Module):
    def __init__(self, output_size):
        super(PolicyNetwork, self).__init__()
        self.l1 = nn.Linear(768, 516)
        self.l2 = nn.Linear(516, 256)
        self.l3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_size)

        nn.init.kaiming_normal_(self.l1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.l2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.l3.weight, mode='fan_out', nonlinearity='relu')   
        nn.init.kaiming_normal_(self.out.weight, mode='fan_out', nonlinearity='relu') 
        
    def forward(self, x):
        x1 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(x1))
        x3 = F.relu(self.l3(x2))
        x_out = self.out(x3)
        return F.softmax(x_out, dim=1)  



def select_action(model, q_embedding, successors, node_to_index):
    probabilities = model(q_embedding).squeeze()  
    mask = torch.zeros_like(probabilities)
    
    indices = torch.tensor([node_to_index[node] for node in successors], dtype=torch.long)
    mask[indices] = 1
    filtered_probabilities = probabilities * mask 
    filtered_probabilities /= filtered_probabilities.sum() 
    m = Categorical(filtered_probabilities)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action, log_prob, filtered_probabilities


if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    with open('database_file_name.pkl', 'rb') as f1:
        database = pickle.load(f1)
    with open('train_file_name.pkl', 'rb') as f2:
        train = pickle.load(f2)
    with open('test_file_name.pkl', 'rb') as f3:
        test = pickle.load(f3)   
   

    with open('database_cot_file_name.pkl', 'rb') as f4:
        database_cot = pickle.load(f4)
    with open('train_cot_file_name.pkl', 'rb') as f5:
        train_cot = pickle.load(f5)
    with open('test_cot_file_name.pkl', 'rb') as f6:
        test_cot = pickle.load(f6)    


    with open('emb_name_1.json', 'r') as f8:
        database_embeddings_dict = json.load(f8)
    database_embeddings_by_name = {item1['case_name']: torch.tensor(item1['embedding']) for item1 in database_embeddings_dict}

    
    with open('emb_name_2.json', 'r') as f9:
        train_embeddings_dict = json.load(f9)
    train_embeddings_by_name = {item2['case_name']: torch.tensor(item2['embedding']) for item2 in train_embeddings_dict}

    
    with open('emb_name_3.json', 'r') as f10:
        test_embeddings_dict = json.load(f10)
    test_embeddings_by_name = {item3['case_name']: torch.tensor(item3['embedding']) for item3 in test_embeddings_dict}    
    print("Test embeddings Loading Completed")

    
    with open('graph.gpickle', 'rb') as f:
        final_graph = pickle.load(f)  

    edges_node = final_graph.edges(data=False)
    final_nd = pd.read_csv("node.csv")
    fact_nodes = final_nd[final_nd["bipartite"]==0].node.tolist()
    all_nodes = final_nd.node.tolist()
    
        
    names_train_beforeshuffer, contents_train_beforeshuffer = get_split_dict(train) 
    names_database, contents_database = get_split_dict(database)
    names_test, contents_test = get_split_dict(test) 

    
    seed = 42
    random.seed(seed)   
    indexes = list(range(len(names_train_beforeshuffer)))
    random.shuffle(indexes)
    names_train = [names_train_beforeshuffer[i] for i in indexes]
    contents_train = [contents_train_beforeshuffer[j] for j in indexes]
    
    node_to_index = {node: idx for idx, node in enumerate(final_graph.nodes())}
    index_to_node = {idx: node for idx, node in enumerate(final_graph.nodes())}


    epochs =  30
    batch_size = 256


    policy_net = PolicyNetwork(output_size=len(all_nodes)).to(device)
    
    

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001, weight_decay=1e-3)


    total_full_batch =  len(contents_train) // batch_size
    total_batches = total_full_batch + (1 if len(contents_train) % batch_size != 0 else 0)


    action_save_all_epoch = []
    max_result = 0
    max_result_distance = 0
    max_result_epoch = 0
    max_result_epoch_distance = 0
    
    for epoch in range(epochs):
        epoch_loss =0
        save_A = []
        flag = 0
        for b_j in tqdm(range(total_batches)):

            start_index = b_j * batch_size
            end_index = min(start_index + batch_size, len(contents_train))
            batch_data = contents_train[start_index:end_index] 
            batch_data_names = names_train[start_index:end_index]

            batch_meteor = 0
            batch_Qscore = 0
            meteor_all_Reward = []

            batch_total_loss = 0         
            
            for k in range(len(batch_data)):
                flag +=1
                actions = []
                instance = batch_data[k]
                current_case_name = batch_data_names[k]

                q_cot = copy.deepcopy(train[current_case_name]["cot"][0])
                a_cot = copy.deepcopy(train[current_case_name]["cot"][1])
                q_cot_embedding = train_embeddings_by_name[current_case_name].to(device)

                one_sample_action = []
                one_sample_action_log = []
            
                action_5 = []
                log_prob_5 = []
                reward_5 = []
                for _ in range(5):
                    successors = list(final_graph.successors(q_cot[-1]))
                    if len(successors) >0:
                        action, log_prob, probabilities = select_action(policy_net, q_cot_embedding, successors,node_to_index)          
                        next_node = index_to_node[action.item()]                      
                        if next_node in a_cot:
                            reward_one_one_step =1
                            q_cot.append(next_node)
                        else:
                            reward_one_one_step = 0
                        action_5.append(action.item())
                        log_prob_5.append(log_prob)
                        reward_5.append(reward_one_one_step)
                    else:
                        break

                losses_onesample = torch.zeros(1, device=device, requires_grad=True)
                for index in range(len(reward_5)):
                    log_probs = torch.stack(log_prob_5)
                    rewards = torch.tensor(reward_5, dtype=torch.float32, device=log_probs.device)
                    case_loss_onestep = -torch.sum(log_probs * rewards)
                    losses_onesample = losses_onesample + case_loss_onestep
                batch_total_loss +=  losses_onesample         

           
            optimizer.zero_grad()
            batch_total_loss.backward()            
            optimizer.step()
            epoch_loss += batch_total_loss

            
            

        print(f"Epoch {epoch}/{epochs}, Total Loss: {epoch_loss.cpu().detach().numpy()}")

        final_path_pre = {}
        sum_reward_epoch = 0
        flag = 0

        for i_test in range(len(names_test)):
            name_now_test = names_test[i_test]
            instance_test = test[name_now_test]

            test_q_cot = copy.deepcopy(instance_test["cot"][0])
            test_a_cot = copy.deepcopy(instance_test["cot"][1])
            test_q_cot_embedding = test_embeddings_by_name[name_now_test].to(device)
                    

            action_5_test = []
            log_prob_5_test = []
            reward_5_test = []
            for _ in range(3):
                successors_test = list(final_graph.successors(test_q_cot[-1]))
                if len(successors_test) >0:
                    action_test, log_prob_test, probabilities_test = select_action(policy_net, test_q_cot_embedding, successors_test,node_to_index)          
                    next_node_test = index_to_node[action_test.item()]
                    test_q_cot.append(next_node_test)
                    if next_node_test in test_a_cot:
                        reward_one_one_step_test =1
                    else:
                        reward_one_one_step_test = 0
                    action_5_test.append(action_test.item())
                    log_prob_5_test.append(log_prob_test)
                    reward_5_test.append(reward_one_one_step)

                else:
                    break
            one_sample_reward = sum(reward_5_test)
            sum_reward_epoch += one_sample_reward
            
            final_path_pre[name_now_test] = test_q_cot
 
        save_path_name = str(epoch)+"_predict_path.pkl"
        with open(save_path_name, 'wb') as f11:
            pickle.dump(final_path_pre, f11)  