
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
# from dataloaders.dataloader_compositional import *


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
env = CompositionGrid(composite_config2)
env.plot_board()


state_model = LearnableEmbedding(device, BATCH_SIZE).to(device)
try:
    state_model.load_state_dict(torch.load(LOWER_STATE_MODEL_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")

action_model = LowerPolicyTrainer(device, BATCH_SIZE, MAX_TIMESTEPS).to(device)
try:
    action_model.load_state_dict(torch.load(LOWER_ACTION_MODEL_PATH))

except:
    print("COULD NOT LOAD ACTION NETWORK")