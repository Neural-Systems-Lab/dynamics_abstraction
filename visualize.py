
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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
BATCH_SIZE = 100
SAMPLES = 1000
MODEL_PATH = "/home/vsathish/core_projects/saved_models/mar_31_run_10.state"

#####################
# Load Data and Model
#####################

data1, data2 = get_transitions()

hypernet_x1, hypernet_y1 = batch_data(data1, BATCH_SIZE)
hypernet_x2, hypernet_y2 = batch_data(data2, BATCH_SIZE)

print(hypernet_x1.shape, hypernet_y2.shape)

hypernet_x1 = torch.from_numpy(hypernet_x1).to(device, dtype=torch.float32)
hypernet_x2 = torch.from_numpy(hypernet_x2).to(device, dtype=torch.float32)
hypernet_y1 = torch.from_numpy(hypernet_y1).to(device, dtype=torch.float32)
hypernet_y2 = torch.from_numpy(hypernet_y2).to(device, dtype=torch.float32)

model = LearnableStory(device, BATCH_SIZE).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")




l1, _, higher1, _ = model(hypernet_x1[0], hypernet_y1[0], eval_mode=True) 
print("Done with env 1")
l2, _, higher2, _ = model(hypernet_x2[0], hypernet_y2[0], eval_mode=True)
print(l1, l2)
print(higher1[-1])

# sys.exit(0)
##################
# TSNE Embeddings
##################

h_concat = np.concatenate([higher1[-1], higher2[-1]], axis=0)
print("######### EMBEDDINGS #########")
# print(h_concat)
print(h_concat.shape, type(h_concat))

tsne_ = TSNE(n_components=2, init='pca', perplexity=20)
h_embed = tsne_.fit_transform(h_concat)
print(h_embed.shape)

pca_ = PCA(n_components=2)
pca_embed = pca_.fit_transform(h_concat)

# Plotting
size = len(h_embed)
print(size)
x1 = h_embed[:int(size/2), 0]
x2 = h_embed[int(size/2):, 0]

y1 = h_embed[:int(size/2), 1]
y2 = h_embed[int(size/2):, 1]

plt.clf()
plt.close()

plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.legend()
plt.title("TSNE of higher latents")
plt.savefig("TSNE_latents.png")


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
plt.title("PCA of higher latents")
plt.savefig("PCA_latent.png")
sys.exit(0)