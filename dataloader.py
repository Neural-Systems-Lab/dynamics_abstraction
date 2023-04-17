import numpy as np
import random

from env import SimpleGridEnvironment
from configs import *
# from state_model_hypernet import MatrixModule



GOALS = [(0, 0), (2, 2), (0, 2), (2, 0)]
STARTS = [(0, 2), (2, 2), (2, 0), (0, 0)]
ACTIONS = {0:[0, 0, 0, 1],
           1:[0, 0, 1, 0],
           2:[0, 1, 0, 0],
           3:[1, 0, 0, 0]
           }

##########################
# Gather Data to train
##########################

'''
Data is of the format : [(state, action, next_state), (state, action, next_state), ...]
'''

TRAJECTORIES = 1000
STEPS = 25

def get_transitions():
    dataset1 = []
    dataset2 = []


    test_env1 = SimpleGridEnvironment(config=config1 , goal=(0, 0), start_states=STARTS.copy())
    test_env2 = SimpleGridEnvironment(config=config2, goal=(2, 2), start_states=STARTS.copy())

    prefix = [0, 1]
    cur_state = test_env1.reset()
    for _ in range(TRAJECTORIES):
        trajectory = []
        for s in range(STEPS):
            action = random.randint(0, 3)
            next_state, reward, end = test_env1.step(action)
            act = ACTIONS[action]
            trajectory.append((np.concatenate((prefix,cur_state)), \
                np.array(act), np.concatenate((prefix, next_state))))
            
            cur_state = next_state

        dataset1.append(trajectory)
        cur_state = test_env1.reset()
        
    prefix = [1, 0]
    cur_state = test_env2.reset()
    for _ in range(TRAJECTORIES):
        trajectory = []
        for s in range(STEPS):
            action = random.randint(0, 3)
            next_state, reward, end = test_env2.step(action)
            act = ACTIONS[action]
            trajectory.append((np.concatenate((prefix, cur_state)), \
                np.array(act), np.concatenate((prefix, next_state))))
            
            cur_state = next_state

        dataset2.append(trajectory)
        cur_state = test_env2.reset()
        
    return dataset1, dataset2


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
    
    