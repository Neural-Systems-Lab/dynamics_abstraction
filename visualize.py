
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

device = torch.device("cuda")
# device = "cpu"
BATCH_SIZE = 100
SAMPLES = 1000
MODEL_PATH = "../saved_models/may_8_run_1.state"

#####################
# Load Data and Model
#####################


data1, data2, data3, data4 = get_transitions()

x1, y1 = batch_data(data1, BATCH_SIZE)
x2, y2 = batch_data(data2, BATCH_SIZE)
x3, y3 = batch_data(data3, BATCH_SIZE)
x4, y4 = batch_data(data4, BATCH_SIZE)

print(x2.shape, y3.shape)

x1 = torch.from_numpy(x1).to(device, dtype=torch.float32)
x2 = torch.from_numpy(x2).to(device, dtype=torch.float32)
x3 = torch.from_numpy(x3).to(device, dtype=torch.float32)
x4 = torch.from_numpy(x4).to(device, dtype=torch.float32)


y1 = torch.from_numpy(y1).to(device, dtype=torch.float32)
y2 = torch.from_numpy(y2).to(device, dtype=torch.float32)
y3 = torch.from_numpy(y3).to(device, dtype=torch.float32)
y4 = torch.from_numpy(y4).to(device, dtype=torch.float32)

####################
# Model
####################

model = LearnableStory(device, BATCH_SIZE).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")


l1, _, higher1 = model(x1[0], y1[0], eval_mode=True) 
l2, _, higher2 = model(x2[0], y2[0], eval_mode=True) 
l3, _, higher3 = model(x3[0], y3[0], eval_mode=True) 
l4, _, higher4 = model(x4[0], y4[0], eval_mode=True) 
print(l1, l2, l3, l4)

higher1 = np.array(higher1)
higher2 = np.array(higher2)
higher3 = np.array(higher3)
higher4 = np.array(higher4)


######################
# Episodic TSNE of r_h
######################

t1 = higher1[:, 0, :]
t2 = higher2[:, 0, :]
t3 = higher3[:, 0, :]
t4 = higher4[:, 0, :]


episodes = np.concatenate([t1, t2, t3, t4], axis=0)
e_tsne = TSNE(n_components=2, init='pca')
e_embed = e_tsne.fit_transform(episodes)

print(episodes.shape, e_embed.shape)

size = len(e_embed)
p1 = int(size/4)
p2 = 2 * int(size/4)
p3 = 3 * int(size/4)

x1 = e_embed[:p1, 0]
x2 = e_embed[p1:p2, 0]
x3 = e_embed[p2:p3, 0]
x4 = e_embed[p3:, 0]

y1 = e_embed[:p1, 1]
y2 = e_embed[p1:p2, 1]
y3 = e_embed[p2:p3, 1]
y4 = e_embed[p3:, 1]


plt.clf()
plt.close()

alpha0 = 0.04

_arr = [x for x in range(len(x1))]
colors1 = []
colors2 = []
colors3 = []
colors4 = []

for x in _arr:
    alpha = alpha0 * (x+1)
    c1 = (0.4, 0.2, 0.8, alpha)
    c2 = (0.8, 0.2, 0.3, alpha)
    c3 = (0.4, 0.8, 0.8, alpha)
    c4 = (0.2, 0.8, 0.3, alpha)
    colors1.append(c1)
    colors2.append(c2)
    colors3.append(c3)
    colors4.append(c4)



plt.scatter(x1, y1, color=colors1, label="Env 1")
plt.scatter(x2, y2, color=colors2, label="Env 2")
plt.scatter(x3, y3, color=colors3, label="Env 3")
plt.scatter(x4, y4, color=colors4, label="Env 4")

plt.plot(x1, y1, linewidth=0.2)
plt.plot(x2, y2, linewidth=0.2)
plt.plot(x3, y3, linewidth=0.2)
plt.plot(x4, y4, linewidth=0.2)

plt.scatter(x1[-1], y1[-1], color="black", s=55)
plt.scatter(x2[-1], y2[-1], color="black", s=55)
plt.scatter(x3[-1], y3[-1], color="black", s=55)
plt.scatter(x4[-1], y4[-1], color="black", s=55)
plt.legend()
plt.title("TSNE of latent converging over an episode - embedding method")
plt.savefig("../plots/episodic_tsne.png")


# sys.exit(0)
##################
# TSNE Embeddings
##################

h_concat = np.concatenate([higher1[-1], higher2[-1], higher3[-1], higher4[-1]], axis=0)
print("######### EMBEDDINGS #########")
print(h_concat.shape, type(h_concat))

tsne_ = TSNE(n_components=2, init='pca', perplexity=20)
h_embed = tsne_.fit_transform(h_concat)
print(h_embed.shape)

pca_ = PCA(n_components=2)
pca_embed = pca_.fit_transform(h_concat)

# Plotting
size = len(h_embed)
p1 = int(size/4)
p2 = 2 * int(size/4)
p3 = 3 * int(size/4)


print(size)
x1 = h_embed[:p1, 0]
x2 = h_embed[p1:p2, 0]
x3 = h_embed[p2:p3, 0]
x4 = h_embed[p3:, 0]

y1 = h_embed[:p1, 1]
y2 = h_embed[p1:p2, 1]
y3 = h_embed[p2:p3, 1]
y4 = h_embed[p3:, 1]

plt.clf()
plt.close()

plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.scatter(x3, y3, label="Env 3")
plt.scatter(x4, y4, label="Env 4")
plt.legend()
plt.title("TSNE of higher latents - embedding method")
plt.savefig("../plots/TSNE_latents.png")

# sys.exit(0)
##################
# PCA Embeddings
##################
size = len(pca_embed)
p1 = int(size/4)
p2 = 2 * int(size/4)
p3 = 3 * int(size/4)

x1 = pca_embed[:p1, 0]
x2 = pca_embed[p1:p2, 0]
x3 = pca_embed[p2:p3, 0]
x4 = pca_embed[p3:, 0]

y1 = pca_embed[:p1, 1]
y2 = pca_embed[p1:p2, 1]
y3 = pca_embed[p2:p3, 1]
y4 = pca_embed[p3:, 1]

plt.clf()
plt.close()

plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.scatter(x3, y3, label="Env 3")
plt.scatter(x4, y4, label="Env 4")
plt.legend()
plt.title("PCA of higher latents - embedding method")
plt.savefig("../plots/PCA_latent.png")
sys.exit(0)