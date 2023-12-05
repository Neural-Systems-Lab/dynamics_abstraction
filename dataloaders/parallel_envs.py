import numpy as np
import random
import sys

import torch.nn as nn
import torch
import torch.nn.functional as F

from environments.env import SimpleGridEnvironment

class ParallelEnvironments():
    '''
    A Vectorized environment interface for learning RL policies
    '''
    def __init__(self, configs, batch_size, device):
        '''
        for each config:    
            define environments
        
        define necessary variables
        '''
        self.configs = configs
        self.batch_size = batch_size
        self.device=device
        self.unique_env_count = 0

        for c in configs:
            for i in range(len(c["abs_actions"])):
                # e = SimpleGridEnvironment(c, c["subgoals"][i], c["abs_actions"][i])
                # self.envs.append(e)
                self.unique_env_count += 1
        
        assert (batch_size%self.unique_env_count == 0)
        self.splits = int(batch_size / self.unique_env_count)
        
        self.parallel_envs = []
        self.higher_actions = []
        for c in configs:
            for i in range(len(c["abs_actions"])):
                for _ in range(self.splits):
                    e = SimpleGridEnvironment(c, c["subgoals"][i], c["abs_actions"][i])
                    self.parallel_envs.append(e)
                    self.higher_actions.append(np.array(c["abs_actions"][i]).flatten())
        
        self.higher_actions = np.array(self.higher_actions)
        print("higher actions shape : ", self.higher_actions.shape)

    def get_higher_actions(self):
        return torch.from_numpy(self.higher_actions).to(self.device, dtype=torch.float32)

    def batch_reset(self):
        '''
        given the list of envs, reset each env
        '''
        cur_states = []
        for env in self.parallel_envs:
            state = env.reset()
            cur_states.append(state)
        cur_states = np.array(cur_states)

        return torch.from_numpy(cur_states).to(self.device, dtype=torch.float32)

    def batch_step(self, batch_action):

        next_states, batch_rewards, batch_mask = [], [], []

        for i in range(len(batch_action)):
            env = self.parallel_envs[i]
            state, reward, mask = env.step(batch_action[i], track_end=True)
            next_states.append(state)
            batch_rewards.append(reward)
            batch_mask.append(mask)
        
        next_states, batch_rewards, batch_mask = np.array(next_states), \
                            np.array(batch_rewards), np.array(batch_mask)

        return torch.tensor(next_states).to(self.device, dtype=torch.float32),\
                torch.tensor(batch_rewards).to(self.device, dtype=torch.float32),\
                torch.tensor(batch_mask).to(self.device, dtype=torch.float32) 
