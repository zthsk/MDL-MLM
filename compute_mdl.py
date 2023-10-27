from transformers import  AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import random
import torch
import re
import gc
import time
import argparse
import json
import os
import math 

start_time = time.time()

#random.seed(1000)
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(1500)


#splits the dataset into different portions based on the fracs
def split_data_into_portions(dataset_train_dataset):
    total_len = len(dataset_train_dataset)
    fractions = fracs
    train_portions = []
    eval_portions = []
    for i in range(len(fractions)):
        train_portions.append(dataset_train_dataset[: max(1,int(fractions[i] * total_len))])
        if i != len(fractions) - 1:
            eval_portions.append(dataset_train_dataset[int(fractions[i] * total_len):
                                                        max(int(fractions[i] * total_len) + 1, int(fractions[i + 1] * total_len))])
    return train_portions[:-1], eval_portions

#tokenize the batch and return the final inputs for the model
def tokenize_batch(train_data, org_data):
    inputs = tokenizer(org_data, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    masked_inputs = tokenizer(train_data, return_tensors='pt', max_length = 256, truncation = True, padding = 'max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()
        
    inputs['input_ids'] = masked_inputs.input_ids.detach().clone()

    return inputs

#finetune the masked language model for predicting masked tokens
def train_mdl(train_loader, org_train_loader, args):
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    epochs = args.epoch
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(zip(train_loader, org_train_loader), leave=True)
        for train_batch, org_batch in loop:
            optim.zero_grad()
            # pull all tensor batches required for training
            inputs = tokenize_batch(train_batch, org_batch) 
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = inputs['labels'].to(device)
            # get the predictions from the model 
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

#test the masked language model to get the logs
def test_mdl(test_loader, org_test_loader):
    model.eval()
    loop = tqdm(zip(test_loader, org_test_loader), leave=True)
    #evaluate data for one epoch
    total_loss = []
    with torch.no_grad():
        for test_batch, org_batch in loop:
            inputs = tokenize_batch(test_batch, org_batch)
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = inputs['labels'].to(device)
    
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss.append(loss.item())
            loop.set_description('Eval: ')
            loop.set_postfix(loss=loss.item())
    return total_loss

#get the appropriate training data for the mlm task
def get_data(task):
    if args.mdl_direction == 'causal':
        if task == 'causal_removed':
            org_data = f'H_filled_{anti_list}.csv'
            mdl_data = f'c_masked_{anti_list}.csv'
        elif task == 'causal_notfinetuned':
            org_data = f'modified_data.csv'
            mdl_data = f'c_masked_{anti_list}.csv'
    elif args.mdl_direction == 'anticausal':
        if task == 'anticausal_removed':
            org_data = f'modified_data.csv'
            mdl_data = f'c_masked_{anti_list}.csv'
        elif task == 'anticausal_notfinetuned':
            org_data = f'H_filled_{anti_list}.csv'
            mdl_data = f'c_masked_{anti_list}.csv'
    
    #load the original train data
    original_df = pd.read_csv(f'{exp_path}/{org_data}')
    original_txt = original_df['reviews'].tolist()

    #mlm_train_data
    mlm_train_df = pd.read_csv(f'{exp_path}/{mdl_data}')
    mlm_train_data = mlm_train_df['reviews'].tolist()
            
    #converting the reviews and class column to a list
    train_datasets, test_datasets = split_data_into_portions(mlm_train_data)
    
    original_train, original_test = split_data_into_portions(original_txt)
    
    return train_datasets, test_datasets, original_train, original_test
    

def create_directory(dir_name):
    try:
        path = os.path.join(exp_path, dir_name)
        os.mkdir(path)
        print("Directory '% s' created" % dir_name)
    except:
        print(f'Directory {dir_name} already exists!!!') 


# ==============================================================
#               Main Iteration of the program
# ==============================================================
#create a parser to parse tha arguments passed
parser = argparse.ArgumentParser()
#defining arguments

parser.add_argument('--epoch',
                    type = int,
                    default = 3,
                    help = 'Number of iterations to train the model')

parser.add_argument('--batch_size',
                    type = int,
                    default = 1,
                    help = 'Size of the batch')

parser.add_argument('--causal_percent',
                    type = str,
                    default = '100',
                    choices = ['100','75', '50', '25', '0'],
                    help = 'percentage of causal words in H')

parser.add_argument('--experiment',
                    type = int,
                    default = 8,
                    choices = list(range(0, 20)),
                    help = 'Experiment number')

parser.add_argument('--anticausal',
                    type = int,
                    default = 1,
                    choices = list(range(1, 4)),
                    help = 'Iteration number')

parser.add_argument('--mdl_direction',
                    type = str,
                    default = 'causal',
                    choices = ['causal','anticausal'],
                    help = 'Direction of the MDL Computation')

#get the arguments
args = parser.parse_args()  
print('\n {} \n'.format(args))        

#define the fraction of datasets
fracs = [0.001,0.002,0.004,0.008,0.016,0.032,0.0625,0.125,0.25,0.5,1]

#define the tasks for MDL computation
causalMDL_task = ['causal_removed', 'causal_notfinetuned']
anticausalMDL_task = ['anticausal_removed', 'anticausal_notfinetuned']

#different list of non-modified words
anti_list = args.anticausal

#specify the name of the experiment and the directory for it to save the results
dir_name = f'{args.causal_percent}_H_{args.experiment}'
splits = dir_name.split('_')

torch.cuda.empty_cache()

#load the word list
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print('Tokenizer Loaded!!!')

exp_path = f'amazon_experiment/exp_{str(args.experiment)}/{args.causal_percent}H'

#diretory to store the mdl values for each task
directory_name = 'mdl_values'
create_directory(directory_name) 

max_length = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('Device: ', device)

if args.mdl_direction == 'causal':
    task_list = causalMDL_task
elif args.mdl_direction =='anticausal':
    task_list = anticausalMDL_task

for i in range(len(task_list)):
    name = task_list[i]
    print('Task Name: ', name)
    
    print(f'Starting the training: Outputs P({name} | remaining words)')

    train_datasets, test_datasets, original_train, original_test = get_data(name)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    timestamp_length = []       #store the length of training data in each timestamp
    timestep_loss = []          #store the total loss for each timestep
    # and move our model over to the selected device
    for timestamp, train_data, test_data, org_train, org_test in zip(fracs[:-1], train_datasets, test_datasets, original_train, original_test):
        sum_loss = 0
        if fracs.index(timestamp) == 0 and i == 0:
            model_path = f'{exp_path}/bert-finetuned_{anti_list}'
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            special_tokens_dict = {'additional_special_tokens': ['[CAUSAL]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))

        elif i == 0:
            index = fracs.index(timestamp)
            prev_model = 'bert-uncased-mlm-'+str(fracs[index-1])
            path = f'{exp_path}/saved_models/{directory_name}/{name}/{prev_model}'
            model = AutoModelForMaskedLM.from_pretrained(path)
            special_tokens_dict = {'additional_special_tokens': ['[CAUSAL]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))
            print('{} model has been loaded locally!!!'.format(prev_model))

        elif i == 1:
            model_path = f'{exp_path}/saved_models/{directory_name}/{task_list[i-1]}/bert-uncased-mlm-0.5'
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            special_tokens_dict = {'additional_special_tokens': ['[CAUSAL]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))
            
        train_loader = DataLoader(train_data,  batch_size = args.batch_size, drop_last = True)
        org_train_loader = DataLoader(org_train,  batch_size = args.batch_size, drop_last = True)

        test_loader = DataLoader(test_data, batch_size = args.batch_size, drop_last = True)
        org_test_loader = DataLoader(org_test,  batch_size = args.batch_size, drop_last = True)
        model.to(device)
        
        print('\n Training the data on timestamp: ', timestamp)
        if i == 0:
            train_mdl(train_loader, org_train_loader, args)
            
        ts_loss = test_mdl(test_loader, org_test_loader)
        sum_loss = sum_loss + sum(ts_loss) #sum up all the loss in the given timestep
        timestep_loss.append(sum_loss)
        timestamp_length.append(len(train_data))
        
        print(f'\n\nSum of loss for timestamp {timestamp}: {sum_loss}\n\n')  
         
        model_name = 'bert-uncased-mlm-'+str(timestamp)
        path = f'{exp_path}/saved_models/{directory_name}/{name}/{model_name}'
        model.save_pretrained(path)
        #model.save_pretrained(path, from_pt=True)   
    
    data = []
    for t, length, loss in zip(fracs[:-1], timestamp_length, timestep_loss):
        data.append({'Timestep': t, 'Train Data': length, 'Loss': loss})
    
    mdl = timestamp_length[0] * math.log2(8000) + sum(timestep_loss)
    data.append({'Online codelength': round(mdl / 1024, 4)})
    # Write the data to a JSON file
    with open(f'{exp_path}/{directory_name}/{name}_{anti_list}.json', 'w') as f:
        json.dump(data, f, indent=4)

end_time = time.time()
total_time = end_time - start_time

hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print("Total runtime: {:.0f} hours {:.0f} minutes {:.2f} seconds".format(hours, minutes, seconds))