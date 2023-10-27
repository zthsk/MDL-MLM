from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import torch
import re
import gc
import time
import argparse
import time
import math

start_time = time.time()
#import sys



def mask_words(reviews):
    #set the exclusion tokens and the masking rate
    exclusion_tokens = ['[CAUSAL]', '[MASK]']
    mask_percentage = random.randint(20,30) / 100

    masked_review = []
    masked_label = []
    
    for text in reviews:
        text = text.replace('[MASK]', '[CAUSAL]')
        masked_label.append(text)
        org_words = text.split()
        masked_words = org_words.copy()

        # Step 2: Identify the positions of the exclusion tokens
        exclusion_indices = [i for i, token in enumerate(org_words) if token in exclusion_tokens]

        # Step 3: Determine the number of words to mask
        total_words = len(org_words) - len(exclusion_indices)
        num_words_to_mask = int(mask_percentage * total_words)

        # Step 4: Randomly select words to mask
        non_exclusion_indices = [i for i in range(len(org_words)) if i not in exclusion_indices]
        words_to_mask_indices = random.sample(non_exclusion_indices, num_words_to_mask)

        # Step 5: Replace selected words with [MASK] token
        for idx in words_to_mask_indices:
            masked_words[idx] = '[MASK] ' * len(tokenizer.tokenize(org_words[idx]))

        #make the complete sentence
        masked_text = ' '.join(masked_words)
        masked_text = re.sub(r'\s+', ' ', masked_text.strip())
        masked_review.append(masked_text)

    #return the masked texts
    return masked_review, masked_label


def mask_tokenizer(datasets):
    masked_datasets, masked_labels =  mask_words(datasets)

    inputs = tokenizer(masked_labels, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    masked_inputs = tokenizer(masked_datasets, return_tensors='pt', max_length = 256, truncation = True, padding = 'max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()
        
    inputs['input_ids'] = masked_inputs.input_ids.detach().clone()

    return inputs


def train_mdl(loader, args):
    # activate training mode
    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=5e-5)
    epochs = args.epoch
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            
            # get the masked inputs from the tokenizer
            inputs = mask_tokenizer(batch) #tokenize the batch and return the final inputs for the model
            
            #load inputs, labels and attention mask
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = inputs['labels'].to(device)
            
            #get the outputs and loss from the model
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            
            #backward propagation and compute gradients
            loss.backward()
            optim.step()
            
            #print the loss and progress of epochs
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

def free_gpu(model):
    model.cpu()
    del model
    gc.collect()
    #torch.cuda.empty_cache()
    
    
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
                    default = 64,
                    help = 'Size of the batch')
                 
parser.add_argument('--causal_percent',
                    type = str,
                    default = '100',
                    choices = ['100','75', '50', '25', '0'],
                    help = 'percentage of causal words in H')
                    
parser.add_argument('--experiment',
                    type = int,
                    default = 0,
                    choices = list(range(0, 20)),
                    help = 'Experiment number')
                    
parser.add_argument('--anticausal',
                    type = int,
                    default = 1,
                    choices = list(range(1, 4)),
                    help = 'Iteration number')

args = parser.parse_args()  
print('\n {} \n'.format(args))        


#initialize the time variable
timestr = time.strftime("%Y-%m-%d-%H:%M:%S")

#initialize the device to use 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('\nDevice used for computation: {} \n'.format(device))

anti_list = args.anticausal

# Set the experiment path
exp_path = f'amazon_experiment/exp_{str(args.experiment)}/{args.causal_percent}H'
    
 
#load the dataset to be used
dataset_name = f'c_masked_{anti_list}.csv'

dataset = pd.read_csv(f'{exp_path}/{dataset_name}')
mdl_train = dataset['reviews'].tolist()

#initialize the tokenizer and bert model for finetuning the MLM model
model_checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

special_tokens_dict = {'additional_special_tokens': ['[CAUSAL]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

print(f'\nFinetuning BERT MLM model with the contexts \n')

#initialize the train data loader
mlmtrain_loader = DataLoader(mdl_train, batch_size = args.batch_size)

#load the model into the device
model.to(device)

#function to finetune the mlm model
train_mdl(mlmtrain_loader, args)

#set path and save the finetuned model
model_name = f'{exp_path}/bert-finetuned_{anti_list}'


model.save_pretrained(model_name)

end_time = time.time()

total_time = end_time - start_time

hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print("\nTotal runtime: {:.0f} hours {:.0f} minutes {:.2f} seconds".format(hours, minutes, seconds))
    
    