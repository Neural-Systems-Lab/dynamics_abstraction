
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F


from environments.composition import CompositionGrid
from environments.pomdp_config import *
from dataloaders.dataloader_compositional import *
from models.embedding_model import LearnableEmbedding


###################
# CONSTANTS
###################

device = torch.device("cuda")
# device = torch.device("cpu")
HYPER_EPOCHS = 50
BATCH_SIZE = 100
WARMUP_EPISODES = 100
LOAD_PATH = "saved_models/pomdp/oct_25_run_1_hypernet.state"
# SAVE_PATH = "../saved_models/pomdp/may_8_run_2.state"

# Define env
env = CompositionGrid(composite_config3)
env.plot_board()

# Get dataset
# dataset = get_transitions(env)
# x, y = batch_data(dataset, BATCH_SIZE)
# print(x.shape, y.shape)

