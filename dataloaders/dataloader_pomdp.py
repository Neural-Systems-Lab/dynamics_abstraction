import numpy as np
import random

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

TRAJECTORIES = 1000
STEPS = 25

configs = [c1, c2]
datasets = []

def get_transitions():

    for config in configs:
        env = SimpleGridEnvironment(config=config, goal=(2, 0))
        env.plot_board()
        dataset = []
        cur_state = env.reset()
        for _ in range(TRAJECTORIES):
            trajectory = []
            for s in range(STEPS):
                action = random.randint(0, 3)
                next_state, reward, end = env.step(action)
                act = ACTIONS[action]
                trajectory.append((cur_state, np.array(act), next_state))
                
                cur_state = next_state

            dataset.append(trajectory)
            cur_state = env.reset()
    
        datasets.append(dataset)
    return datasets


def batch_data(dataset, batch_size):
    print("here : ", len(dataset), len(dataset[0]))
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

        for time in range(STEPS):
        
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