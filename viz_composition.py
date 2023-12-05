
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
import matplotlib.lines as mlines

from environments.composition import CompositionGrid
from dataloaders.dataloader import get_transitions as gt
from dataloaders.dataloader import batch_data as bd
from dataloaders.dataloader_compositional import get_transitions, batch_data

from models.embedding_model import LearnableEmbedding
from environments.pomdp_config import *
# from models.learnable_story import LearnableStory
##############
# CONSTANTS
##############

device = torch.device("mps")
# device = "cpu"
BATCH_SIZE = 100
TIMESTEPS = 25
MODEL_PATH = "/Users/vsathish/Documents/Quals/saved_models/pomdp/oct_25_run_1_embedding.state"

###################################
# Load Compositional Data and Model
###################################

env = CompositionGrid(composite_config3)
# env.plot_board()
dataset = get_transitions(env)
x1, y1 = batch_data(dataset, BATCH_SIZE)

print(x1.shape, y1.shape)

x1 = torch.from_numpy(x1).to(device, dtype=torch.float32)
y1 = torch.from_numpy(y1).to(device, dtype=torch.float32)

####################
# Model
####################
timesteps = 1000
# model = LearnableEmbedding(device, BATCH_SIZE, timesteps).to(device)
model = LearnableEmbedding(device, BATCH_SIZE).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## NOPE #######################")

# sys.exit(0)
l1, _, higher1 = model(x1[0], y1[0], eval_mode=True) 
# sys.exit(0)
print(l1)

higher1 = np.array(higher1)
print("here 1 : ", higher1.shape, len(higher1))

######################
# Episodic TSNE of r_h
######################

env.print_episode(0)
print("higher states : ", higher1.shape)
t1 = higher1[:, 0, :]

colors_ = []
map_ = {0:"red", 1:"green", 2:"blue", 3:"orange"}

for s, _, _, _ in env.historic_data[0]:
    
    colors_.append(map_[env.higher_states[s]])

print(colors_)

# episodes = np.concatenate([t1, t2, t3, t4], axis=0)
episodes = t1
e_tsne = TSNE(n_components=2, init='pca')
e_embed = e_tsne.fit_transform(episodes)

# for e in episodes:
    

x1 = e_embed[:,0]
y1 = e_embed[:,1]
print("TSNE embedding : ", e_embed.shape)

# plt.clf()
# plt.close()


plt.scatter(x1, y1, color=colors_)
plt.plot(x1, y1, linewidth=0.2)
plt.scatter(x1[-1], y1[-1], color="black", s=55)

plt.title("TSNE of latent converging over an episode - embedding method")
plt.savefig("../plots/compositional/episodic_tsne.png")


sys.exit(0)
##################
# TSNE Embeddings
##################

print("Last few Latents : ", higher1[-1].shape)
tsne_ = TSNE(n_components=2, init='pca', perplexity=20)
h_embed = tsne_.fit_transform(higher1[-1])
print(h_embed.shape)

pca_ = PCA(n_components=2)
pca_embed = pca_.fit_transform(higher1[-1])
print("### VARIANCE EXPLAINED ###")
print(pca_.explained_variance_)
print(pca_.explained_variance_ratio_)


# Plotting

plt.clf()
plt.close()

plt.scatter(h_embed[:,0], h_embed[:,1], label="Composite Env")

plt.legend()
plt.title("TSNE of compositional env higher latents - embedding method")
plt.savefig("../plots/composition/TSNE_latents.png")

# sys.exit(0)
##################
# PCA Embeddings
##################

plt.clf()
plt.close()

plt.scatter(pca_embed[:,0], pca_embed[:,1])

plt.legend()
plt.title("PCA of higher latents - embedding method")
plt.savefig("../plots/composition/PCA_latent.png")
sys.exit(0)