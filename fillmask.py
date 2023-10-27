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

def contain_mask_word(sentence):
    return '[MASK]' in sentence


def replace_mask_tokens(sentence, best_guess):
    words = sentence.split()
    mask_indices = [index for index, word in enumerate(words) if word == "[MASK]"]
    if len(mask_indices) != len(best_guess):
        print(f'Length of mask words: {len(mask_indices)}')
        print(f'Length of predicted words: {len(best_guess)}')

        print(type(best_guess))

        import sys
        sys.exit()
    for index, pred_word in zip(mask_indices, best_guess):
        words[index] = pred_word
    
    return ' '.join(words)



def get_prediction(test_batch):
    
    inputs = tokenizer(test_batch, return_tensors='pt', max_length=256, truncation=True, padding='max_length')

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    masked_position = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position ]

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    last_hidden_state = output[0].squeeze()

    list_of_list =[]
    for index,mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=5, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        list_of_list.append(words)
    
    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess+" "+j[0]
        
    return best_guess.split()


def test_mdl(test_loader):
    model.eval()
    mask_filled_sentence = []
    loop = tqdm(test_loader, leave=True)
    #evaluate data for one epoch
    #all_predictions = []
    with torch.no_grad():
        for test_batch in loop:
            for sentence in test_batch:
                if contain_mask_word(sentence) == True:
                    predicted_blanks = get_prediction(sentence)
                    new_sentence = replace_mask_tokens(sentence, predicted_blanks)
                    mask_filled_sentence.append(new_sentence)
                else:
                    mask_filled_sentence.append(sentence)

    return mask_filled_sentence

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
                    default = 128,
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
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('\nDevice used for computation: {} \n'.format(device))

anti_list = args.anticausal

exp_path = f'amazon_experiment/exp_{str(args.experiment)}/{args.causal_percent}H'
    
 
dataset_name = f'c_masked_{anti_list}.csv'
save_name = f'H_filled_{anti_list}.csv'

dataset = pd.read_csv(f'{exp_path}/{dataset_name}')
test_data = dataset['reviews'].tolist()

#initialize the tokenizer and bert model for finetuning the MLM model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_path = f'{exp_path}/bert-finetuned_{anti_list}'
model = AutoModelForMaskedLM.from_pretrained(model_path)

#initialize the train data loader
test_loader = DataLoader(test_data, batch_size = args.batch_size)

#load the model into the device
model.to(device)

#function to finetune the mlm model
final_sentence = test_mdl(test_loader)

filled_sentence_df = pd.DataFrame(final_sentence, columns = ['reviews'])
filled_sentence_df.to_csv(f'{exp_path}/{save_name}', index = None)


