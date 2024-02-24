
import sys
import random
import numpy as np
import torch

from environments.pomdp_config import *
from environments.env import SimpleGridEnvironment

##########################
# Gather Data to train
##########################

'''
Data is of the format : [(state, action, next_state), (state, action, next_state), ...]
'''

ACTIONS = {0:[0, 0, 0, 1],
           1:[0, 0, 1, 0],
           2:[0, 1, 0, 0],
           3:[1, 0, 0, 0]
           }

TRAIN = 600
TEST = 200
TRAJECTORIES = TRAIN + TEST
DEFAULT_STEPS = 25

configs = [c1, c2, c3, c4, c5]
# configs = [c1, c3, c4]
# print(configs)
def get_transitions(num_envs, timesteps):
    datasets = []
    for config in configs[:num_envs]:
        env = SimpleGridEnvironment(config=config, goal=(2, 0))
        env.plot_board()
        dataset = []
        cur_state = env.reset()
        for _ in range(TRAJECTORIES):
            trajectory = []
            for s in range(timesteps):
                action = random.randint(0, 3)
                next_state, reward, end = env.step(action)
                # pomdp_state = env.get_pomdp_state()
                act = ACTIONS[action]

                # Add noise to the state
                # Flip a random bit in the state with a probability of 0.1
                if np.random.uniform() < 0.1:
                    idx0 = np.random.choice(np.where(cur_state == 0)[0])
                    cur_state[idx0] = 1
                cur_state = cur_state + np.random.normal(0, 0.2, cur_state.shape)

                trajectory.append((cur_state, np.array(act), next_state))
                
                cur_state = next_state
            # sys.exit(0)
            dataset.append(trajectory)
            cur_state = env.reset()
    
        datasets.append(dataset)
    return datasets


def batch_data(dataset, batch_size, timesteps):
    print("here (Num Episodes, Num Steps): ", len(dataset), len(dataset[0]))
    '''
    len(dataset) % Batch == 0
    Input = n * (Batch, timesteps, in_dims)
    Output = n * (Batch, timesteps, out_dims)
    '''
    batch_input = []
    batch_output = []
    
    start=0
    while start < len(dataset):
        # print(start)
        inputs = []
        outputs = []

        for time in range(timesteps):
        
            input_t = []
            output_t = []
            
            for i in range(start, start+batch_size):

                ip = np.concatenate((dataset[i][time][0], dataset[i][time][1]))
                op = dataset[i][time][2]
                
                input_t.append(ip)
                output_t.append(op)
        
            inputs.append(input_t)
            outputs.append(output_t)
        
        batch_input.append(inputs)
        batch_output.append(outputs)
    
        start = start+batch_size
    
    batch_input = np.array(batch_input)
    batch_output = np.array(batch_output)
    
    return batch_input, batch_output


def generate_data(batch_size, device, num_envs=2, timesteps=DEFAULT_STEPS):

    data = get_transitions(num_envs, timesteps)
    train_test_env_splits = []

    # For each environment batch and split the data into train and test
    for d in data:
        train, test = d[:TRAIN], d[TRAIN:]
        x_train, y_train = batch_data(train, batch_size, timesteps)
        x_test, y_test = batch_data(test, batch_size, timesteps)
        
        x_train = torch.from_numpy(x_train).to(device, dtype=torch.float32)
        y_train = torch.from_numpy(y_train).to(device, dtype=torch.float32)
        x_test = torch.from_numpy(x_test).to(device, dtype=torch.float32)
        y_test = torch.from_numpy(y_test).to(device, dtype=torch.float32)

        train_test_env_splits.append((x_train, y_train, x_test, y_test))

    assert len(train_test_env_splits) == num_envs

    return train_test_env_splits
