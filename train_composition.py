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


###################
# CONSTANTS
###################

device = torch.device("cuda")
# Data Generation Constants
COMPOSITION_CONFIG = composite_config3
BASE_CONFIGS = [c1, c3, c4]
NUM_SAMPLES = 1000


HYPER_EPOCHS = 50
BATCH_SIZE_STATE = 100
BATCH_SIZE = 1
POLICY_TIMESTEPS = 20
INFERENCE_TIMESTEPS = 20
LOWER_STATE_MODEL_PATH = "/mmfs1/gscratch/rao/vsathish/quals/saved_models/state_network/feb_9_run_1_embedding.state"
LOWER_ACTION_MODEL_PATH = "../saved_models/action_network/feb_11_run_2_action_embedding.state"
HIGHER_STATE_MODEL_PATH = "../saved_models/higher_state_network/feb_12_run_1_higher_state_"+COMPOSITION_CONFIG["name"]+".state"
COMPOSITION_DATA_PATH = "/mmfs1/gscratch/rao/vsathish/quals/saved_models/composition_data/"

# Define env
env = CompositionGrid(COMPOSITION_CONFIG)
env.plot_board(name="composition3")


sys.exit(0)
# Define lower state model
lower_state_model = LearnableEmbedding(device, BATCH_SIZE_STATE, timesteps=INFERENCE_TIMESTEPS).to(device)
try:
    lower_state_model.load_state_dict(torch.load(LOWER_STATE_MODEL_PATH))
    print("################## LOAD SUCCESS #################")
except:
    print("################## NOPE #######################")

# Define lower action model
lower_action_model = LowerPolicyTrainer(device, BATCH_SIZE, POLICY_TIMESTEPS).to(device)
try:
    lower_action_model.load_state_dict(torch.load(LOWER_ACTION_MODEL_PATH))
except:
    print("COULD NOT LOAD ACTION NETWORK")

# Define higher state model
higher_state_model = AbstractStateNetwork(4, 16, COMPOSITION_CONFIG["num_blocks"]).to(device)
# try:
#     higher_state_model.load_state_dict(torch.load(HIGHER_STATE_MODEL_PATH))
# except:
#     print("COULD NOT LOAD HIGHER STATE NETWORK")



# Define data generator
# lower_state_model.eval()
lower_action_model.eval()
data_gen = AbstractStateDataGenerator(COMPOSITION_CONFIG, BASE_CONFIGS, \
            lower_state_model, lower_action_model, INFERENCE_TIMESTEPS, BATCH_SIZE_STATE, device)
x_train, y_train = data_gen.generate_data(NUM_SAMPLES, env)

print("################## DATA GENERATED #################")
print("################## Printing errors #################")
print(data_gen.error_counter)
print(data_gen.state_error_counter)
print(x_train.shape, y_train.shape)


###################
# SAVE THE DATA
###################

torch.save(x_train, os.path.join(COMPOSITION_DATA_PATH, "x_train_"+COMPOSITION_CONFIG["name"]+".pt"))
torch.save(y_train, os.path.join(COMPOSITION_DATA_PATH, "y_train_"+COMPOSITION_CONFIG["name"]+".pt"))

# Define the optimizer
optimizer = optim.Adam(higher_state_model.parameters(), lr=0.0001)

# Define the loss function
loss_fn = F.mse_loss

# Define the number of epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    higher_state_model.train()
    
    # Reset the optimizer
    optimizer.zero_grad()
    
    # Forward pass
    outputs = higher_state_model(x_train)
    
    # Compute the loss
    loss = loss_fn(outputs, y_train)
    
    # Backward pass
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print the loss for each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
# Save the trained model to HIGHER_STATE_MODEL_PATH
torch.save(higher_state_model.state_dict(), HIGHER_STATE_MODEL_PATH)
