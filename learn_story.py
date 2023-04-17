
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from dataloader import *
from models.learnable_story import LearnableStory

###################
# CONSTANTS
###################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HYPER_EPOCHS = 100
BATCH_SIZE = 100
WARMUP_EPISODES = 100
LOAD_PATH = "../saved_models/apr_10_run_6.state"
SAVE_PATH = "../saved_models/apr_10_run_6.state"
#########################################
# Training a Hypernet Modulated Network
#########################################

# Transforming data

data1, data2 = get_transitions()

hypernet_x1, hypernet_y1 = batch_data(data1, BATCH_SIZE)
hypernet_x2, hypernet_y2 = batch_data(data2, BATCH_SIZE)

print(hypernet_x1.shape, hypernet_y2.shape)

hypernet_x1 = torch.from_numpy(hypernet_x1).to(device, dtype=torch.float32)
hypernet_x2 = torch.from_numpy(hypernet_x2).to(device, dtype=torch.float32)
hypernet_y1 = torch.from_numpy(hypernet_y1).to(device, dtype=torch.float32)
hypernet_y2 = torch.from_numpy(hypernet_y2).to(device, dtype=torch.float32)


#######################
# MODEL LOAD and TRAIN
#######################

model = LearnableStory(device, BATCH_SIZE).to(device)

try:
    model.load_state_dict(torch.load(LOAD_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")

hyper_optim = torch.optim.Adam(model.hypernet.parameters(), model.hyper_lr)
temporal_optim = torch.optim.Adam(model.temporal.parameters(), model.temporal_lr)
train_loss = []
print("Everything cool till now")


for epochs in range(HYPER_EPOCHS):
    epoch_loss = []
    print("Epoch : ", epochs)
    
    for i in range(len(hypernet_x1)):
        
        # Ignore predicted and inferred states during training
        l1 = model(hypernet_x1[i], hypernet_y1[i]) 
    
        l1.backward()
        hyper_optim.step()
        if epochs < WARMUP_EPISODES:    
            temporal_optim.step()

        hyper_optim.zero_grad()
        temporal_optim.zero_grad()
        
        l2 = model(hypernet_x2[i], hypernet_y2[i])
 
        l2.backward()
        hyper_optim.step()
        if epochs < WARMUP_EPISODES:  
            temporal_optim.step() 
        
        print("i = ", i, "loss1 = ", l1.detach().cpu().numpy())
        print("i = ", i, "loss2 = ", l2.detach().cpu().numpy())
        # print("Everything cool inside model train")
        
        epoch_loss.append((l1+l2).detach().cpu().numpy())
    
    print("Mean Loss : ", np.mean(epoch_loss))
    train_loss.append(np.mean(epoch_loss))

    if epochs % 10 == 0:
        print("Saving Checkpoint ... ")
        torch.save(model.state_dict(), SAVE_PATH)
        
torch.save(model.state_dict(), SAVE_PATH)

################
# Plot Learning
################

plt.clf()
plt.close()

plt.plot(train_loss, label="Hypernet")
plt.legend()
plt.savefig("loss.png")

sys.exit(0)
