
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
HYPER_EPOCHS = 50
BATCH_SIZE = 1
MAX_TIMESTEPS = 20
LOWER_STATE_MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/pomdp/oct_25_run_1_embedding.state"
LOWER_ACTION_MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/action_network/dec_6_run_1_embedding.state"
SAVE_PATH = "/Users/vsathish/Documents/Quals/saved_models/pomdp/oct_25_run_1_embedding.state"
COMPOSITION_CONFIG = composite_config2

# Define env
env = CompositionGrid(COMPOSITION_CONFIG)
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

higher_state_model = AbstractStateNetwork(9, COMPOSITION_CONFIG["num_blocks"])
env.reset()
print(env.state, env.get_higher_token())

# Generate data for the state network
datagen = AbstractStateDataGenerator(COMPOSITION_CONFIG)
higher_state_data = datagen.generate_data()