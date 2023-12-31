import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.composition import CompositionGrid
from environments.pomdp_config import *

class AbstractStateNetwork(nn.Module):
    def __init__(self, state_space, global_token_size):
        super().__init__()

        self.state_space = state_space
        self.token_size = global_token_size
        self.output_size = state_space

        self.model = nn.Sequential(
            nn.Linear(state_space+global_token_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_space+global_token_size),
            nn.ReLU()
        )
    
    def forward(self, observations, actions):
        
        # Expects flat observations and actions
        inputs = torch.cat((observations, actions), dim=1)
        outputs = self.model(inputs)

        return outputs


class AbstractStateDataGenerator():
    def __init__(self, config, base_configs, lower_state_model, lower_action_model, device):
        self.config = config
        self.state_model = lower_state_model
        self.action_model = lower_action_model
        self.device = device
        self.centers = np.array(config["centers"])
        self.num_bases = len(self.centers)
        self.env = CompositionGrid(config)
        self.base_configs = base_configs
        
    def reset_state(self, times=50):
        self.env.reset()
        # If the agent is in one of the subgoal state, reset.
        if len(self.env.get_higher_token()) > 1:
            if times == 0:
                print("Failed after max tries")
                return -1
            
            return self.reset(times-1)
    
        return 0

    def generate_data(self, num_samples):

        x_train = [] # [(abstract_state.unique_token), (abstract_action.relative_token)]
        y_train = [] # [(next_abstract_state.unique_token)]
        for sample in range(num_samples):
            self.reset_state()
            current_state, higher_index = self.get_abstract_state()
            random_action = self.get_abstract_action(higher_index)
            next_state = self.step(random_action)
            print(current_state, random_action)
            x_train.append(torch.cat((current_state, random_action), dim=0))
            y_train.append(next_state)
        
        x_train = torch.stack(x_train).to(self.device)
        y_train = torch.stack(y_train).to(self.device)
        return x_train, y_train


    def get_abstract_action(self, higher_index):
        base_config = self.base_configs[higher_index]
        higher_action = np.random.choice(base_config["abs_actions"])
        # How do i get the relative action from the higher action?

    def get_abstract_state(self):
        # Infer abstract state from the lower state network
        # Return abstract state + unique_token
        action_sequence = [0, 1, 2, 3, 0, 1, 2, 3] # Walking in circles
        inference_data_x = []
        inference_data_y = []
        print(self.env.get_pomdp_state())
        cur_state = self.env.get_pomdp_state()
        print("init state : ", self.env.state)
        for action in action_sequence:
            next_state, _, _ = self.env.step(action)
            action_ = np.zeros((4))
            action_[action] = 1
            inference_data_x.append(np.concatenate((cur_state, action_), axis=0))
            inference_data_y.append(next_state)
            cur_state = next_state
        
        inference_data_x = torch.from_numpy(np.array(inference_data_x))
        inference_data_y = torch.from_numpy(np.array(inference_data_y))

        inference_data_x = torch.unsqueeze(inference_data_x.to(self.device, dtype=torch.float32), axis=1)
        inference_data_y = torch.unsqueeze(inference_data_y.to(self.device, dtype=torch.float32), axis=1)

        print(inference_data_x.shape, inference_data_y.shape)

        # Use this data to infer the abstract stat
        loss, _, higher = self.state_model(inference_data_x, inference_data_y, eval_mode=True)
        
        # Min of distances
        center_index = np.argmin(np.linalg.norm(np.expand_dims(higher[-1], axis=0) - self.centers, axis=1))
        higher_state = self.centers[center_index]

        print("Concatenated distances : ", np.linalg.norm(np.expand_dims(higher[-1], axis=0) - self.centers, axis=1))
        print("Final state : ", self.env.state)

        # Concat with token of the state
        unique_token = self.env.get_higher_token()
        if len(unique_token) > 1:
            print("Something is wrong")
            sys.exit(0)

        print("Unique token : ", unique_token)
        higher_state = np.concatenate((higher_state, unique_token[0]), axis=0)
        print("higher state : ", higher_state)
        return torch.from_numpy(higher_state).to(self.device, dtype=torch.float32), center_index

    def step(self, abstract_action):
        pass
