import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    def forward(self, observations, global_token):
        
        inputs = torch.cat((observations, global_token), dim=1)
        outputs = self.model(inputs)

        return outputs


class AbstractDataGenerator():
    def __init__():
        pass
    
