import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environments.composition import CompositionGrid
from environments.pomdp_config import *

class AbstractStateNetwork(nn.Module):
    def __init__(self, state_space, action_space, global_token_size):
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.token_size = global_token_size
        self.output_size = state_space

        self.model = nn.Sequential(
            nn.Linear(state_space+global_token_size+action_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_space+global_token_size),
            # nn.ReLU()
        )
    
    def forward(self, inputs):
        
        # Expects flat observations and actions
        # inputs = torch.cat((observations, actions), dim=1)
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
        
        self.base_configs = base_configs
        
    def reset_state(self, env, goal=None, times=50):
        env.reset(goal=goal)
        # If the agent is in one of the subgoal state, reset.
        if len(env.get_higher_token()) > 1:
            if times == 0:
                print("Failed after max tries")
                return -1
            return self.reset_state(env, goal, times-1)
    
        return 0

    def generate_data(self, num_samples, env=None):
        # self.env = CompositionGrid(self.config)
        if env==None:
            env = CompositionGrid(self.config)

        x_train = [] # [(abstract_state.unique_token), (abstract_action.relative_token)]
        y_train = [] # [(next_abstract_state.unique_token)]
        for sample in range(num_samples):
            print(f"############## SAMPLE {sample} ##############")
            # Reset to a non subgoal state
            self.reset_state(env)
            
            current_state, higher_index = self.get_abstract_state(env)
            abstract_action, subgoal = self.get_abstract_action(higher_index)
            next_state, higher_index = self.step(abstract_action, subgoal, env)
            
            print(current_state, abstract_action, next_state)
            # sys.exit(0)
            x_train.append(torch.cat((current_state, abstract_action), dim=0))
            y_train.append(next_state)
        
        x_train = torch.stack(x_train).to(self.device)
        y_train = torch.stack(y_train).to(self.device)
        return x_train, y_train


    def get_abstract_action(self, higher_index):
        
        base_config = self.base_configs[higher_index]
        higher_action_idx = np.random.choice(len(base_config["abs_actions"]))
        higher_action = np.array(base_config["abs_actions"][higher_action_idx])
        
        # Get the relative action
        relative_action_ = base_config["teleports"][higher_action_idx][0]
        relative_action = np.eye(4)[relative_action_]

        # Concatenate the two and convert to tensor
        # abstract_action = np.concatenate((higher_action, relative_action), axis=0)
        abstract_action = torch.from_numpy(higher_action).to(self.device, dtype=torch.float32).flatten()
        # relative_action = torch.from_numpy(relative_action).to(self.device, dtype=torch.float32)
        print("Running policy for subgoal : ", higher_action_idx, base_config["subgoals"][higher_action_idx])
        
        return abstract_action, relative_action

    def get_abstract_state(self, env):
        print("################## BEGIN ABS STATE ##################")
        # Infer abstract state from the lower state network
        # Return abstract state + unique_token
        action_sequence = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] # Walking in circles
        inference_data_x = []
        inference_data_y = []
        # print(self.env.get_pomdp_state())
        cur_state = env.get_pomdp_state()
        print("init state : ", env.state)
        record_init_state = env.state
        for action in action_sequence:
            next_state, _, _ = env.step(action, record_step=False)
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
        print("Center index : ", center_index)
        # print("Concatenated distances : ", np.linalg.norm(np.expand_dims(higher[-1], axis=0) - self.centers, axis=1))

        # Go back to original state
        env.state = record_init_state
        print("Final state : ", env.state)

        # Concat with token of the state
        unique_token = env.get_higher_token()
        if len(unique_token) > 1:
            print("Something is wrong")
            sys.exit(0)

        # print("Unique token : ", unique_token)
        higher_state = np.concatenate((higher_state, unique_token[0]), axis=0)
        print("higher state : ", higher_state)
        print("################## DONE WITH ABS STATE ##################")
        return torch.from_numpy(higher_state).to(self.device, dtype=torch.float32), center_index

    def step(self, abstract_action, teleport_, env):
        # First run the policy with the abstract action
        env, actions, rewards, end = self.action_model.execute_policy(env, torch.flatten(abstract_action))

        # Then take an additional step with the relative action
        print("Relative action : ", teleport_)
        relative_action = np.argmax(teleport_)
        env.step(relative_action)
        # Finally invoke the get_abstract_state function on the new state
        return self.get_abstract_state(env)
