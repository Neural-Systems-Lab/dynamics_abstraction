
from math import e
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
from models.abstract_state_network import AbstractStateNetwork, AbstractStateDataGenerator
# from dataloaders.dataloader_compositional import *


###################
# CONSTANTS
###################

device = torch.device("mps")
COMPOSITION_CONFIG = composite_config2
BASE_CONFIGS = [c1, c2]
HYPER_EPOCHS = 50
BATCH_SIZE = 1
MAX_TIMESTEPS = 20
LOWER_STATE_MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/pomdp/oct_25_run_1_embedding.state"
LOWER_ACTION_MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/action_network/dec_6_run_1_embedding.state"
HIGHER_STATE_MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/state_network/jan_4_higher_state_composition2.state"

'''
MPC planning:
1. Identify the starting higher state S_T by random walk
2. Use S_T to extract higher actions A_T allowed in that state
3. Use S_T and A_T to plan with higher state model
4. Recurse for k steps, k being a hyperparameter
5. Use the lower state model to plan for the last step
'''


# Define env
env = CompositionGrid(composite_config2)
env.plot_board()


#############################################
lower_state_model = LearnableEmbedding(device, BATCH_SIZE).to(device)
try:
    lower_state_model.load_state_dict(torch.load(LOWER_STATE_MODEL_PATH))
    print("###### STATE NETWORK LOADED SUCCESSFULLY ######")

except:
    print("COULD NOT LOAD ACTION NETWORK")

#############################################

action_model = LowerPolicyTrainer(device, BATCH_SIZE, MAX_TIMESTEPS).to(device)
try:
    action_model.load_state_dict(torch.load(LOWER_ACTION_MODEL_PATH))
    print("###### ACTION NETWORK LOADED SUCCESSFULLY ######")

except:
    print("COULD NOT LOAD ACTION NETWORK")

#############################################
higher_state_model = AbstractStateNetwork(4, 16, COMPOSITION_CONFIG["num_blocks"]).to(device)
higher_state_identifier = AbstractStateDataGenerator(COMPOSITION_CONFIG, BASE_CONFIGS, \
                                                    lower_state_model, action_model, device)
try:
    higher_state_model.load_state_dict(torch.load(HIGHER_STATE_MODEL_PATH))
    print("###### HIGHER STATE NETWORK LOADED SUCCESSFULLY ######")
except:
    print("COULD NOT LOAD HIGHER STATE NETWORK")

#############################################
    
