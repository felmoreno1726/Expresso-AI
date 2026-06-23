import os
import random
import subprocess
import numpy as np
#from sklearn.model_selection import train_test_split


dataset_ids = [str(i) for i in range(32)]

SUBJECT_POOLS_BY_GENDER = {
    'MALE_+'   :   ['0', '1', '2', '3', '7', '9', '10', '14', '15'], 
    'FEMALE_+' :   ['4', '5', '6', '8', '11', '12', '13'], 
    'MALE_-'   :   ['22', '25'], 
    'FEMALE_-' :   ['17', '18', '19', '20', '21', '23', '24', '26', '27', '28', '29', '30', '31']
}

SUBJECT_POOLS = {
    '+' : [i for i in range(16)], 
    '-' : [i for i in range(16, 32)]
}

def train_test_val_split(subject_ids, seed):
    training_ids, testing_ids = train_test_split(subject_ids, test_size=0.1, random_state=seed)
    training_ids, validation_ids = train_test_split(training_ids, test_size=0.1, random_state=1)
    return training_ids, validation_ids, testing_ids

def dataset_split(pools):
    train, val, test = [], [], []
    for label, pool in pools.items():
        training_ids, validation_ids, testing_ids = train_test_val_split(pool, seed=random.randint(0, 2020))
        train += training_ids
        val += validation_ids
        test += testing_ids
    return train, val, test

def format_participant_ids(ids):
    formatted_ids = str(ids)[1:-1].replace(',', ' ')
    print('formatted_ids: ', formatted_ids)
    return  formatted_ids

if __name__ == "__main__":
    num_experiments = 100
    for experiment_id in range(num_experiments):
        print(f"Running experiment {experiment_id}")
        epochs = random.choice([128])
        lr = random.choice([1e-4, 4e-4, 7e-4])
        weight_decay = random.choice([0, 1e-4, 5e-5, 1e-5])
        momentum = random.choice([0.5, 0.9])
        #window_size = random.choice([256])#Try 512
        #input_dilation = random.choice([16])#Try 8, 16, 32
        window_size, input_dilation = random.choice([(128, 8), (256, 16)])
        overlap = random.choice([0.4, 0.6, 0.7])
        pretrained_dataset = random.choice(['K'])
        model_name = random.choice(['r2p1d'])
        model_depth = random.choice([50])
        #regularize
        noise_std = random.choice([1.0, 1.5, 2.0])
        #run experiment
        cmd = f"python3 scripts/face_main.py --train --test  \
        --model_name={model_name} \
        --model_depth={model_depth} \
        --pretrained_dataset={pretrained_dataset} \
        --epochs={epochs} \
        --weight_decay={weight_decay}  \
        --momentum={momentum} \
        --window_size={window_size} \
        --input_dilation={input_dilation} \
        --overlap={overlap} \
        --noise_std={noise_std} \
        --lr={lr} \
        "
        subprocess.call(cmd, shell=True)

