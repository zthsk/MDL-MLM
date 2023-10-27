import os
import shutil
import argparse

def clear_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)



parser = argparse.ArgumentParser()
parser.add_argument('--experiment',
                    type = int,
                    default = 0,
                    choices = list(range(0, 20)),
                    help = 'Experiment number')

parser.add_argument('--causal_percent',
                    type = str,
                    default = '100',
                    choices = ['100','75', '50', '25', '0'],
                    help = 'percentage of causal words in H')    

parser.add_argument('--fill_type',
                    type = str,
                    default = 'causal_h',
                    choices = ['causal_h','anticausal_hprime'],
                    help = 'Causal Words to choose')

                    
args = parser.parse_args()

directory_name = 'mdl_values'
# Directory path to be cleared
directory_path = f'amazon_experiment/exp_{args.experiment}/{args.causal_percent}H/saved_models/{directory_name}'
#directory_path_2 = f'amazon_experiment/swapped_0_15000/exp_{args.experiment}/100H/saved_models/anticausal_direction'

# Call the function to clear the directory
clear_directory(directory_path)
