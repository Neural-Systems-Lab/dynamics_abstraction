import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F

from environments.pomdp_config import *
from models.action_network import LowerPolicyTrainer
from dataloaders.parallel_envs import ParallelEnvironments

# Global constants
EPOCHS = 100
BATCH_SIZE = 40
MAX_TIMESTEPS = 25
HYPER_LR = 0.005
POLICY_LR = 0.001
CRITIC_LR = 0.005
LOAD_PATH = "../saved_models/action_network/dec_3_run_1_embedding.state"
SAVE_PATH = "../saved_models/action_network/dec_3_run_1_embedding.state"
SAVE_FILES = "../plots/action_network/"
device = torch.device("mps")

# Environments to use
env_configs = [c1, c2]

data_handler = ParallelEnvironments(env_configs, BATCH_SIZE, device)
model = LowerPolicyTrainer(data_handler, device, BATCH_SIZE, MAX_TIMESTEPS).to(device)

hypernet_optimizer = torch.optim.Adam(model.hypernet.parameters(), lr=HYPER_LR)
policy_optimizer = torch.optim.Adam(model.policy.parameters(), lr=POLICY_LR)
critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=CRITIC_LR)

def plot_(rewards):
    plt.plot(range(0, len(rewards)), rewards)
    plt.title("Average Episodic Rewards")
    plt.savefig(SAVE_FILES+"rewards.png")
    plt.clf()


for epoch in range(EPOCHS):
    print(f"################ EPOCH {epoch} ####################")
    actor_loss, critic_loss = model()
    print(actor_loss, critic_loss)

    hypernet_optimizer.zero_grad()
    policy_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    loss = actor_loss + critic_loss
    loss.backward()
    hypernet_optimizer.step()
    policy_optimizer.step()
    critic_optimizer.step()
    
    if (epoch+1) % 5 == 0:
        plot_(np.squeeze(model.epoch_rewards))


plot_(np.squeeze(model.epoch_rewards))