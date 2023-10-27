from transformers import BertTokenizer
import json
from tqdm import tqdm
import pandas as pd
import re
import os
from nltk.corpus import stopwords
import random
import math 



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def _tokenize(seq):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys

def get_new_changes(changes, sentences)   :
    new_changes = []
    for change in changes:
        if len(change[1].split()) == len(change[2].split()):
            new_changes.append(change)
        
    return new_changes
        

def get_index_mappings(changes, sentences):
    idx_list = []
    mappings = {}
    for item in changes:
        idx_list.append(item[0])
        org_word = item[2]
        adv_word = item[1]
        mappings[org_word] = adv_word   
    return idx_list, mappings


def get_index_and_mappings(indexes, mappings):
    causal_indexes = []
    causal_mappings = []

    anti_indexes_1 = []
    anti_indexes_2 = []
    anti_indexes_3 = []
    
    for index_list, mapping_dict in zip(indexes, mappings):
        if len(index_list) == 0:
            causal_indexes.append([])
            causal_mappings.append({})

            anti_indexes_1.append([])
            anti_indexes_2.append([])
            anti_indexes_3.append([])
            

        else:
            n = math.floor(len(index_list) * 0.3)
            c_index = random.sample(index_list, n)
            causal_indexes.append(c_index)
            candidates = [idx for idx in index_list if idx not in c_index]

            a_index_1 = random.sample(candidates, n)
            a_index_2 = random.sample(candidates, n)
            a_index_3 = random.sample(candidates, n)

            anti_indexes_1.append(a_index_1)
            anti_indexes_2.append(a_index_2)
            anti_indexes_3.append(a_index_3)

            causal_mappings.append({key: value for key, value in mapping_dict.items() if key in c_index })


    
    anti_indexes = (anti_indexes_1, anti_indexes_2, anti_indexes_3)
    
    
    return causal_indexes, causal_mappings, anti_indexes

def find_modified_words(original, modified):
    mappings = []
    indexes = []
    
    for org, mod in zip(original, modified):
        org_words = org.split()
        mod_words = mod.split()

        modified_indices = []
        modified_mapping = {}
        
        try:
            for i in range(len(mod_words)):
                if mod_words[i] != org_words[i]:
                    modified_indices.append(i)
                    modified_mapping[i] = (org_words[i], mod_words[i])
        except:
            print(original)
            print(modified)
        
        mappings.append(modified_mapping)
        indexes.append(modified_indices)
        
    return indexes, mappings

    
def mask_words(review, mask_indices, mask_token):
    words = review.split()
    replaced_words = [mask_token * len(tokenizer.tokenize(word))  if i in mask_indices else word for i, word in enumerate(words)]
    masked_text = ' '.join(replaced_words)
    return masked_text


def get_removed_masked_data(masked_data, masked_list):
    removed_data = []
    removed_masked_data = []
    ant_masked = []
    loader = tqdm(zip(masked_data, masked_list), desc = 'Getting masked and removed data: ')
    for masked_text, indexes in loader:
        removed_txt = masked_text.replace('[MASK]', '[CAUSAL]')
        removed_data.append(re.sub(r'\s+', ' ', removed_txt.strip()))
        
        ant_masked_txt = mask_words(masked_text, indexes, mask_token = '[ANT] ')
        ant_masked.append(ant_masked_txt)
        c_removed_text = ant_masked_txt.replace('[MASK]', '[CAUSAl]')
        
        removed_masked_txt = c_removed_text.replace('[ANT]', '[MASK]')
        removed_masked_data.append(re.sub(r'\s+', ' ', removed_masked_txt.strip()))    
    return removed_data, removed_masked_data



def anticausal_tokens_index(dataset, final_causal_indexes):      
    anticausal_index_tokens = []
    loader = tqdm(zip(dataset, final_causal_indexes), desc = 'Get the anticausal indexes: ')
    for text, index in loader:
        tokens = text.split()
        tokens_with_indexes = [(i, token) for i, token in enumerate(tokens) if token.lower() not in stopwords and i not in index and len(token)>2]
        #print(tokens_with_indexes)
        num_tokens_to_select = len(index) #make no of causal and anticausal tokens similar
        try:
            anticausal_index_tokens.append(random.sample(tokens_with_indexes, k=num_tokens_to_select))
        except:
            print()
            print(text)
            print(len(text.split()))
            print(len(index))
            print(len(tokens_with_indexes))
            print(tokens_with_indexes)
            import sys
            sys.exit()
    
    anticausal_indexes = [[index for index, _ in sublist] for sublist in anticausal_index_tokens]
    return anticausal_indexes
        

def get_masked_percentage(attacked_data, org_indexes, causal_indexes, anti_indexes, percentage):
    causal_masked = []
    anticausal_masked = []
    causal_masked_index = []
    loader = tqdm(zip(attacked_data, org_indexes, causal_indexes, anti_indexes), desc = f'Getting masked dataset: {percentage}')
    for modified, org_idx, mod_index, anti_index in loader:

        if len(mod_index) != 0:
            
            a_masked = mask_words(modified, anti_index, mask_token = '[MASK] ')
            anticausal_masked.append(a_masked)
            
            if percentage == 0:
                
                candidate_idx = [idx for idx in org_idx if idx not in mod_index + anti_index]
                try:
                    causal_index = random.sample(candidate_idx, len(mod_index))
                except:
                    print('Candidates Length: ', len(candidate_idx))
                    print('Causal Index Length: ', len(mod_index))
                    print(modified)
                    continue
                    causal_index = random.sample(candidate_idx, len(mod_index) -1 )
                causal_masked_index.append(causal_index)
                c_masked = mask_words(modified, causal_index, mask_token = '[MASK] ')
                causal_masked.append(c_masked)
                
            elif percentage == 100:
                causal_index = mod_index
                causal_masked_index.append(causal_index)
                c_masked = mask_words(modified, causal_index, mask_token = '[MASK] ')
                causal_masked.append(c_masked)
                
            else:
                causal_index = random.sample(mod_index, int(percentage/100 * len(mod_index)) )
                candidate_idx = [idx for idx in org_idx if idx not in mod_index + anti_index]
                n = len(mod_index) - len(causal_index)
                causal_index = causal_index + random.sample(candidate_idx, n)
                causal_masked_index.append(causal_index)
                c_masked = mask_words(modified, causal_index, mask_token = '[MASK] ')
                causal_masked.append(c_masked)                   
        else:
            causal_masked.append(modified)
            anticausal_masked.append(modified)
            causal_masked_index.append(mod_index)
    
    final_data = (causal_masked, anticausal_masked, causal_masked_index)
    
    return final_data


def get_final_attacked_data(org_data, causal_mappings, causal_indexes):
    final_attacked = []
    
    for org, maps, indexes in tqdm(zip(org_data, causal_mappings, causal_indexes), desc = 'Final modified data: '):

        org_words = org.split()
        atk_words = org_words.copy()
        for i in range(len(org_words)):
            if i in indexes:
                w_org, w_mod = maps[i]
                atk_words[i] = w_mod
        
        final_attacked.append(' '.join(atk_words))
    
    return final_attacked
            
def get_adv_sentence(subtokens, idx_list, mappings):
    adv_sentence = []
    
    new_idx_list = idx_list #random.sample(idx_list, int(0.65 * len(idx_list)))

    for index in range(len(subtokens)):
        i = 0
        if index in new_idx_list:
            word = []
            word.append(subtokens[index])
            
            i = index + 1
            while i < len(subtokens) and subtokens[i].startswith("##"):
                word.append(subtokens[i][2:])
                i += 1
            
            org_word = ''.join(word)
            adv_word = mappings.get(org_word, org_word)
            
            adv_sentence.append(adv_word)
        else:
            word = []
            word.append(subtokens[index])
            i = index + 1
            while i < len(subtokens) and subtokens[i].startswith("##"):
                word.append(subtokens[i][2:])
                i += 1
            org_word = ''.join(word)
            adv_sentence.append(org_word)
            
    final_adv = [word for word in adv_sentence if not word.startswith("##")]
    return ' '.join(final_adv)



def format_text(sentence):
    sentence = re.sub(r'\b_+\b', '', sentence)
    sentence = re.sub(r'_+', ' ', sentence)
    return sentence

def tokenize_reviews(file_path, num_data):
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        loader = tqdm(data, desc = 'Getting adverserial text: ')
        adv_text = []
        org_text = []
        mappings_all = []
        counter = 0# counter to track the processed entries
        for entry in loader:
            
            if counter >= num_data:
                break
            counter += 1
        
            sentence = entry['seq_a'] 
            sentence = format_text(sentence)
            
            changes = entry['changes']
            org_text.append(sentence)
            
            new_changes = get_new_changes(changes, sentence)
            if len(new_changes) == 0 or len(new_changes) == 1 or len(new_changes) == 2:
                adv_text.append(sentence)
                mappings_all.append([])
                continue
            else:
                idx_list, mappings = get_index_mappings(new_changes, sentence)
                tokens, subtokens, keys = _tokenize(sentence)
                adv_sen = get_adv_sentence(subtokens, idx_list, mappings)
                adv_text.append(adv_sen)
                mappings_all.append(mappings)
                
            

    return adv_text, org_text, mappings

def save_list_as_txt(my_list, filename):
    with open(filename, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')

    print("List saved as", filename)         

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
        
def create_save_df(i, j, c_masked, a_masked, c_removed, a_removed, c_masked_a_removed, a_masked_c_removed):
    save_dir = f'{dir_name}/{i}H'
    create_directory(save_dir)
    
    c_masked_DF = pd.DataFrame(c_masked, columns = ['reviews'])
    a_masked_DF = pd.DataFrame(a_masked, columns = ['reviews'])
    
    c_removed_DF = pd.DataFrame(c_removed, columns = ['reviews'])
    a_removed_DF = pd.DataFrame(a_removed, columns = ['reviews'])
    
    c_masked_a_removed_DF = pd.DataFrame(c_masked_a_removed, columns = ['reviews'])
    a_masked_c_removed_DF = pd.DataFrame(a_masked_c_removed, columns = ['reviews'])
       
    c_masked_DF.to_csv(f'{save_dir}/c_masked_{j}.csv', index = None)
    a_masked_DF.to_csv(f'{save_dir}/a_masked_{j}.csv', index = None)
    
    c_removed_DF.to_csv(f'{save_dir}/c_removed_{j}.csv', index = None)
    a_removed_DF.to_csv(f'{save_dir}/a_removed_{j}.csv', index = None)
    
    c_masked_a_removed_DF.to_csv(f'{save_dir}/c_masked_a_removed_{j}.csv', index = None)
    a_masked_c_removed_DF.to_csv(f'{save_dir}/a_masked_c_removed_{j}.csv', index = None)
    

seeds = [1111, 2222, 3333, 4444, 5555]#, 6666, 7777, 8888, 9999, 1500]#, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 123, 524]
 #load the stopwords
stopwords = stopwords.words('english')
    
for i,seed in enumerate(seeds):
    print('\n')
    random.seed(seed)
    #name of the directory to save the data and load the json file
    dir_name = f'amazon_data/h_hprime_atn_ltst/exp_{i}'
    print(f' Directory:  {dir_name}')
              
   
    #load the logs json
    json_file_path = 'amazon_data/amazon_swapped.json'
    
    #extract the data from the json file
    attacked_data, org_data, mappings = tokenize_reviews(json_file_path, num_data = 20000)
    
    mod_indexes, mod_mappings = find_modified_words(org_data, attacked_data)
       
    causal_indexes, causal_mappings, anti_indexes = get_index_and_mappings(mod_indexes, mod_mappings)
    
    final_attacked_data = get_final_attacked_data(org_data, causal_mappings, causal_indexes)
    
    anti_indexes_1, anti_indexes_2, anti_indexes_3 = anti_indexes

    #get new causal and anticausal masked with % of modified words masked
    
    output_100_1 = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_1, percentage = 100)
    output_100_2 = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_2, percentage = 100)
    output_100_3 = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_3, percentage = 100)
    
    
    output_75_1  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_1, percentage = 75)
    output_75_2  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_2, percentage = 75)
    output_75_3  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_3, percentage = 75)
    
    output_50_1  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_1, percentage = 50)
    output_50_2  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_2, percentage = 50)
    output_50_3  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_3, percentage = 50)
    
    output_25_1  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_1, percentage = 25)
    output_25_2  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_2, percentage = 25)
    output_25_3  = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_3, percentage = 25)
    
    output_0_1   = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_1, percentage = 0)
    output_0_2   = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_2, percentage = 0)
    output_0_3   = get_masked_percentage(final_attacked_data, mod_indexes, causal_indexes, anti_indexes_3, percentage = 0)
    
    
    causal_masked_100_1 , anticausal_masked_100_1 , causal_index_100_1   = output_100_1
    causal_masked_100_2 , anticausal_masked_100_2 , causal_index_100_2   = output_100_2
    causal_masked_100_3 , anticausal_masked_100_3 , causal_index_100_3   = output_100_3
    
    causal_masked_75_1  , anticausal_masked_75_1  , causal_index_75_1    = output_75_1
    causal_masked_75_2  , anticausal_masked_75_2  , causal_index_75_2    = output_75_2
    causal_masked_75_3  , anticausal_masked_75_3  , causal_index_75_3    = output_75_3
    
    causal_masked_50_1 , anticausal_masked_50_1  , causal_index_50_1    = output_50_1
    causal_masked_50_2 , anticausal_masked_50_2  , causal_index_50_2    = output_50_2
    causal_masked_50_3 , anticausal_masked_50_3  , causal_index_50_3    = output_50_3
    
    causal_masked_25_1  , anticausal_masked_25_1  , causal_index_25_1    = output_25_1
    causal_masked_25_2  , anticausal_masked_25_2  , causal_index_25_2    = output_25_2
    causal_masked_25_3  , anticausal_masked_25_3  , causal_index_25_3    = output_25_3
    
    causal_masked_0_1   , anticausal_masked_0_1   , causal_index_0_1     = output_0_1
    causal_masked_0_2   , anticausal_masked_0_2   , causal_index_0_2     = output_0_2
    causal_masked_0_3   , anticausal_masked_0_3   , causal_index_0_3     = output_0_3
    
    #list of all the data
    h_values = [100, 75, 50, 25, 0]
    causal_masked_data_1 = [causal_masked_100_1, causal_masked_75_1, causal_masked_50_1, causal_masked_25_1, causal_masked_0_1]
    causal_masked_data_2 = [causal_masked_100_2, causal_masked_75_2, causal_masked_50_2, causal_masked_25_2, causal_masked_0_2]
    causal_masked_data_3 = [causal_masked_100_3, causal_masked_75_3, causal_masked_50_3, causal_masked_25_3, causal_masked_0_3]
    
    causal_indexes_1 = [causal_index_100_1, causal_index_75_1, causal_index_50_1, causal_index_25_1, causal_index_0_1]
    causal_indexes_2 = [causal_index_100_2, causal_index_75_2, causal_index_50_2, causal_index_25_2, causal_index_0_2]
    causal_indexes_3 = [causal_index_100_3, causal_index_75_3, causal_index_50_3, causal_index_25_3, causal_index_0_3]
    
    
    anticausal_masked_data_1 = [anticausal_masked_100_1, anticausal_masked_75_1, anticausal_masked_50_1, anticausal_masked_25_1, anticausal_masked_0_1]
    anticausal_masked_data_2 = [anticausal_masked_100_2, anticausal_masked_75_2, anticausal_masked_50_2, anticausal_masked_25_2, anticausal_masked_0_2]
    anticausal_masked_data_3 = [anticausal_masked_100_3, anticausal_masked_75_3, anticausal_masked_50_3, anticausal_masked_25_3, anticausal_masked_0_3]
    
    anticausal_indexes_1 = [anti_indexes_1, anti_indexes_1, anti_indexes_1, anti_indexes_1, anti_indexes_1]
    anticausal_indexes_2 = [anti_indexes_2, anti_indexes_2, anti_indexes_2, anti_indexes_2, anti_indexes_2]
    anticausal_indexes_3 = [anti_indexes_3, anti_indexes_3, anti_indexes_3, anti_indexes_3, anti_indexes_3]
    
    zipped_1 = zip(h_values, causal_masked_data_1, causal_indexes_1, anticausal_masked_data_1, anticausal_indexes_1)
    zipped_2 = zip(h_values, causal_masked_data_2, causal_indexes_2, anticausal_masked_data_2, anticausal_indexes_2)
    zipped_3 = zip(h_values, causal_masked_data_3, causal_indexes_3, anticausal_masked_data_3, anticausal_indexes_3)
    
    zipped_data = zip([1,2,3], [zipped_1, zipped_2, zipped_3])
    
    for j, zipped in zipped_data:
        for i, c_masked, c_index, a_masked, a_index in zipped:
            c_removed, a_masked_c_removed = get_removed_masked_data(c_masked, a_index)
            a_removed, c_masked_a_removed = get_removed_masked_data(a_masked, c_index)
        
            create_save_df(i, j, c_masked, a_masked, c_removed, a_removed, c_masked_a_removed, a_masked_c_removed)
    #make the dataframe for original and attacked data
    org_df = pd.DataFrame(org_data, columns=['reviews'])
    final_atk_df = pd.DataFrame(final_attacked_data, columns=['reviews'])
    atk_df = pd.DataFrame(attacked_data, columns=['reviews'])
       
    #save the dataframes as csv files
    for i in h_values:
        org_df.to_csv(f'{dir_name}/{i}H/original_data.csv', index=False)
        atk_df.to_csv(f'{dir_name}/{i}H/attacked_data.csv', index=False)
        final_atk_df.to_csv(f'{dir_name}/{i}H/final_attacked_data.csv', index=False)
    
    print('\n')




