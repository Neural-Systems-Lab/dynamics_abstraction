
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
HYPER_EPOCHS = 5
BATCH_SIZE = 100
LR1 = 0.01
LR2 = 0.05

#########################################
# Training a Hypernet Modulated Network
#########################################

# Transforming data

data1, data2 = get_transitions()

hypernet_x1, hypernet_y1 = batch_data(data1, BATCH_SIZE)
hypernet_x2, hypernet_y2 = batch_data(data2, BATCH_SIZE)

print(hypernet_x1.shape, hypernet_y2.shape)

hypernet_x1 = torch.from_numpy(hypernet_x1).to(device, dtype=torch.float64)
hypernet_x2 = torch.from_numpy(hypernet_x2).to(device, dtype=torch.float64)
hypernet_y1 = torch.from_numpy(hypernet_y1).to(device, dtype=torch.float64)
hypernet_y2 = torch.from_numpy(hypernet_y2).to(device, dtype=torch.float64)


#######################
# MODEL LOAD and TRAIN
#######################

model = LearnableStory(device).to(device)

try:
    model.load_state_dict(torch.load("../saved_models/batch_inference_0330.state"))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")

hyper_optim = torch.optim.Adam(model.hypernet.parameters(), model.hyper_lr)
temporal_optim = torch.optim.Adam(model.temporal.parameters(), model.temporal_lr)
loss4 = []
print("Everything cool till now")


for epochs in range(HYPER_EPOCHS):
    epoch_loss = []
    print("Epoch : ", epochs)
    
    for i in range(len(hypernet_x1)):
        hyper_optim.zero_grad()
        temporal_optim.zero_grad()
        
        # Ignore predicted and inferred states during training
        l1, _, _ = model(hypernet_x1[i], hypernet_y1[i]) 
        l2, _, _ = model(hypernet_x2[i], hypernet_y2[i])
        loss = (l1+l2)/2
        
        print("EVerything cool inside model train")
        sys.exit(0)
        
        loss.backward()
        hyper_optim.step()
        temporal_optim.step()
        
        print("i = ", i, "loss = ", loss.detach().cpu().numpy())
        
        epoch_loss.append(loss.detach().cpu().numpy())
        
    loss4.append(np.mean(epoch_loss))

torch.save(model.state_dict(), "../saved_models/samplewise_inference_0322.state")


##################
# TSNE Embeddings
##################

higher1_list = []
higher2_list = []

for i in range(len(hypernet_data1)):
    l1, _, higher1 = model(hypernet_data1[i]) 
    l2, _, higher2 = model(hypernet_data2[i])
    print(i, higher1)
    higher1_list.append(higher1)
    higher2_list.append(higher2)

h1 = np.array(higher1_list).astype('float32')
h2 = np.array(higher2_list).astype('float32')

h_ = np.concatenate([h1, h2], axis=0)
print("######### EMBEDDINGS #########")
print(h_)
print(h_.shape, type(h_))

tsne_ = TSNE(n_components=2, init='random', perplexity=5)
h_embed = tsne_.fit_transform(h_)
print(h_embed.shape)

pca_ = PCA(n_components=2)
pca_embed = pca_.fit_transform(h_)

# Plotting
size = len(h_embed)
x1 = h_embed[:int(size/2), 0]
x2 = h_embed[int(size/2):, 0]

y1 = h_embed[:int(size/2), 1]
y2 = h_embed[int(size/2):, 1]

plt.clf()
plt.close()

plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.legend()

plt.savefig("higher_latents.png")

##################
# PCA Embeddings
##################
x1 = pca_embed[:int(size/2), 0]
x2 = pca_embed[int(size/2):, 0]

y1 = pca_embed[:int(size/2), 1]
y2 = pca_embed[int(size/2):, 1]

plt.clf()
plt.close()

plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.legend()

plt.savefig("pca_higher.png")