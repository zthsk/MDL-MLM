# Implications of Minimum Description Length for Adversarial Attack in NLP

## Dependencies
- Python 3.10
- [PyTorch](https://github.com/pytorch/pytorch) 1.13.1
- [transformers](https://github.com/huggingface/transformers) 4.24.0

## Obtaining the adversarial attack dataset
Before computing the mdl, we need to:
- Train a classifier on the data (*amazon_classifier*)
- Obtain the perturbed data using [BERT-ATTACK](https://github.com/LinyangLee/BERT-Attack).

### To generate the adversarial data based on *amazon_classifier*, run
```
cd adversarial_attack
```
```
python bertattack.py --data_path amazon_original.tsv --mlm_path bert-base-uncased --tgt_path models/amazon_classifier --use_sim_mat 1 --output_dir amazon_logs.tsv --num_label 2 --use_bpe 1 --k 48 --start 0 --end 1000 --threshold_pred_score 0
```
Follow [BERT-ATTACK](https://github.com/LinyangLee/BERT-Attack) for the usage of different arguments. 

## Computing MDL
- Finetune the mlm model and obtain a finetuned model for each datasets, run 
    ```
    python finetune_mlm.py --epoch 3 --batch_size 64 --causal_percent 100 --experiment 1 --anticausal 1
    ```
 - Use the finetuned model to generate tokens for masked datasets
    ```
    python fillmask.py --epoch 3 --batch_size 64 --causal_percent 100 --experiment 1 --anticausal 1
    ```
- Compute the mdl for each causal direction
    ```
    python compute_mdl.py --epoch 2 --batch_size 20 --causal_percent 100 --experiment 1 --anticausal 1 --mdl_direction causal
    ```
- Delete the saved models during mdl computation
    ```
    python clear_saved_models.py --causal_percent 100 --experiment 1
    ```

* --epoch : No. of epoch to train the model
* --batch_size: Size of each batch
* --causal_percent: Different variations of H [0, 25, 50, 75, 100]
* --experiment: We have different variations of datasets with different modified tokens
* --anticausal: For each modified tokens, we have 3 different lists of original tokens
* --mdl_direction: Direction to compute the MDL [causal, anticausal]

### The computed MDL can be found *mdl_values* directory inside *amazon_data* directory. Just traverse through the experiment numbers and the variations of H to get the *mdl_values* for each variations.
    
