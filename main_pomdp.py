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

# from environments.env import SimpleGridEnvironment
# from environments.pomdp_config import *
from dataloaders.dataloader_pomdp import *
from models.learnable_story import LearnableStory
from models.embedding_model import LearnableEmbedding
device = torch.device("mps")

HYPER_EPOCHS = 50
BATCH_SIZE = 100
WARMUP_EPISODES = 100
LOAD_PATH = "../saved_models/pomdp/oct_25_run_1_embedding.state"
SAVE_PATH = "../saved_models/pomdp/oct_25_run_1_embedding.state"

data1, data2 = get_transitions()
x1, y1 = batch_data(data1, BATCH_SIZE)
x2, y2 = batch_data(data2, BATCH_SIZE)

print(x1.shape, y2.shape)

x1 = torch.from_numpy(x1).to(device, dtype=torch.float32)
x2 = torch.from_numpy(x2).to(device, dtype=torch.float32)
y1 = torch.from_numpy(y1).to(device, dtype=torch.float32)
y2 = torch.from_numpy(y2).to(device, dtype=torch.float32)
# model = LearnableStory(device, BATCH_SIZE).to(device)
model = LearnableEmbedding(device, BATCH_SIZE).to(device)

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
    
    for i in range(len(x1)):
        
        # Ignore predicted and inferred states during training
        hyper_optim.zero_grad()
        temporal_optim.zero_grad()
        
        l1 = model(x1[i], y1[i]) 
        l2 = model(x2[i], y2[i])

        
        loss = l1+l2
        loss.backward()
        hyper_optim.step()
        temporal_optim.step()
        
        print("i = ", i, "loss = ", loss.detach().cpu().numpy())
        epoch_loss.append(loss.detach().cpu().numpy())

    
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
plt.savefig("../plots/loss.png")

sys.exit(0)
