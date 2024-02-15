
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

from dataloaders.dataloader_pomdp import *
from models.hypernet_model import LearnableHypernet

###################
# CONSTANTS
###################

device = torch.device("cuda")
# device = torch.device("cpu")
HYPER_EPOCHS = 50
BATCH_SIZE = 100
WARMUP_EPISODES = 100
LOAD_PATH = "../saved_models/state_network/jan30_run_1_hypernet.state"
SAVE_PATH = "../saved_models/state_network/jan30_run_2_hypernet.state"
#########################################
# Training a Hypernet Modulated Network
#########################################

# Get the data

data1, data2 = generate_data(BATCH_SIZE, device)
train_x1, train_y1, test_x1, test_y1 = data1
train_x2, train_y2, test_x2, test_y2 = data2

#######################
# MODEL LOAD and TRAIN
#######################

model = LearnableHypernet(device, BATCH_SIZE).to(device)

try:
    model.load_state_dict(torch.load(LOAD_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")

hyper_optim = torch.optim.Adam(model.hypernet.parameters(), model.hyper_lr)
temporal_optim = torch.optim.SGD(model.temporal.parameters(), model.temporal_lr)
train_loss = []
print("Everything cool till now")


for epochs in range(HYPER_EPOCHS):
    epoch_loss = []
    print("Epoch : ", epochs)
    
    for i in range(len(train_x1)):
        
        # Ignore predicted and inferred states during training
        hyper_optim.zero_grad()
        temporal_optim.zero_grad()
        
        l1, cluster_centers1 = model(train_x1[i], train_y1[i]) 
        l2, cluster_centers2 = model(train_x2[i], train_y2[i])
        # l3 = model(x3[i], y3[i])
        # l4 = model(x4[i], y4[i])
        
        loss = l1+l2
        loss.backward()
        hyper_optim.step()
        temporal_optim.step()
        
        print("i = ", i, "epoch loss = ", loss.detach().cpu().numpy())
        epoch_loss.append(loss.detach().cpu().numpy())
        
    
    print("Mean Loss Per step: ", np.mean(epoch_loss)/(STEPS*2))
    train_loss.append(np.mean(epoch_loss))

    if epochs % 10 == 0 and epochs != 0:
        print("Saving Checkpoint ... ")
        torch.save(model.state_dict(), SAVE_PATH)
        plt.clf()
        plt.close()

        plt.plot(train_loss, label="Hypernet")
        plt.legend()
        plt.savefig("../plots/loss/hypernet_train_loss.png")
        
torch.save(model.state_dict(), SAVE_PATH)

################
# Plot Learning
################

plt.clf()
plt.close()

plt.plot(train_loss, label="Hypernet")
plt.legend()
plt.savefig("../plots/loss/hypernet_train_loss.png")

sys.exit(0)
