
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F


from environments.composition import CompositionGrid
from models.embedding_model import LearnableEmbedding
from models.action_network import LowerPolicyTrainer
from environments.pomdp_config import *
from models.higher_state_network import AbstractStateNetwork
# from dataloaders.dataloader_compositional import *

'''
Higher level world model

(higher_state S + higher_token, higher_action) -> (higher_state_next + higher_token_next)

procedure:

'''


###################
# CONSTANTS
###################

device = torch.device("mps")
HYPER_EPOCHS = 50
BATCH_SIZE = 1
MAX_TIMESTEPS = 20
LOWER_STATE_MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/pomdp/oct_25_run_1_embedding.state"
LOWER_ACTION_MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/action_network/dec_6_run_1_embedding.state"
SAVE_PATH = "/Users/vsathish/Documents/Quals/saved_models/pomdp/oct_25_run_1_embedding.state"

# Define env
env = CompositionGrid(composite_config1)
env.plot_board()


lower_state_model = LearnableEmbedding(device, BATCH_SIZE).to(device)
try:
    lower_state_model.load_state_dict(torch.load(LOWER_STATE_MODEL_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")

lower_action_model = LowerPolicyTrainer(device, BATCH_SIZE, MAX_TIMESTEPS).to(device)
try:
    lower_action_model.load_state_dict(torch.load(LOWER_ACTION_MODEL_PATH))

except:
    print("COULD NOT LOAD ACTION NETWORK")


higher_state_model = AbstractStateNetwork(9, composite_config1["num_blocks"])
env.reset()
print(env.get_higher_token())


