
from math import e
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.apc_planner import AbstractPlanner

from environments.composition import CompositionGrid
from models.embedding_model import LearnableEmbedding
from models.action_network import LowerPolicyTrainer
from environments.pomdp_config import *
from models.abstract_state_network import AbstractStateNetwork, AbstractStateDataGenerator
# from dataloaders.dataloader_compositional import *


###################
# CONSTANTS
###################

device = torch.device("cuda")
COMPOSITION_CONFIG = composite_config2
BASE_CONFIGS = [c1, c2]
HYPER_EPOCHS = 50
BATCH_SIZE = 1
INFERENCE_TIMESTEPS = 1
MAX_TIMESTEPS = 10
PLANNING_HORIZON = 1

LOWER_STATE_MODEL_PATH = "../saved_models/state_network/jan_24_run_3_embedding.state"
LOWER_ACTION_MODEL_PATH = "../saved_models/action_network/jan_23_run_1_action_embedding.state"
HIGHER_STATE_MODEL_PATH = "../saved_models/state_network/jan_24_run_1_higher_state_"+COMPOSITION_CONFIG["name"]+".state"

'''
MPC planning:
1. Identify the starting higher state S_T by random walk
2. Use S_T to extract higher actions A_T allowed in that state
3. Use S_T and A_T to plan with higher state model
4. Recurse for k steps, k being a hyperparameter
5. Use the lower state model to plan for the last step
'''




#############################################
lower_state_model = LearnableEmbedding(device, BATCH_SIZE, timesteps=INFERENCE_TIMESTEPS).to(device)
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
higher_state_util = AbstractStateDataGenerator(COMPOSITION_CONFIG, BASE_CONFIGS, \
                                                    lower_state_model, action_model, device)
try:
    higher_state_model.load_state_dict(torch.load(HIGHER_STATE_MODEL_PATH))
    print("###### HIGHER STATE NETWORK LOADED SUCCESSFULLY ######")
except:
    print("COULD NOT LOAD HIGHER STATE NETWORK")

#############################################
# Load the Abstract Planner
# The Abstract State Network must be specific 
# to the composed environment
#############################################    

planner = AbstractPlanner(COMPOSITION_CONFIG, BASE_CONFIGS, higher_state_model, \
                        higher_state_util, lower_state_model, action_model, \
                        PLANNING_HORIZON, device)

# Define environment
env = CompositionGrid(composite_config2)
# env.plot_board()
START_STATE = (2, 0)
GOALS = [(2, 0), (2, 4), (2, 8), (0, 9), (4, 9), (0, 10), (4, 10)]

# for state in env.valid_pos:
#     if len(env.higher_states[state]) > 1:
#         GOALS.append(state)

print(GOALS)
# sys.exit(0)
counters = []
for goal in GOALS:
    print("############## GOAL : ", goal, " ##############")
    # Reset to a non-subgoal state
    higher_state_util.reset_state(env, START_STATE, goal)

    # Plan
    steps, counter = planner.abstract_plan(env)
    print(counter)
    counters.append(counter)
    for s in steps:
        print(s[0])
    print(goal)


print(counters)