# %%
import os
import sys
import json
sys.path.insert(1, "/gscratch/rao/vsathish/quals/dynamics_abstraction")
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import gauss2d
from models.embedding_model import LearnableEmbedding, INFER_PARAMS
from dataloaders.dataloader_pomdp import generate_data, configs
from environments.env import SimpleGridEnvironment
from environments.composition import CompositionGrid
from environments.pomdp_config import *
# from dataloaders.dataloader_pomdp import TRAIN, TEST, STEPS

#############################################
# CONSTANTS - Define before running this cell
#############################################

device = torch.device("cuda")
BATCH_SIZE = 200
SAMPLES = 1000
NUM_ENVS = 5
TIMESTEPS = 15
PLOT_SAVE_PATH = "/gscratch/rao/vsathish/quals/plots/state_network/"
MODEL_PATH = "/mmfs1/gscratch/rao/vsathish/quals/saved_models/state_network/feb_13_run_4_embedding.state"
# PARAMS_PATH = "/mmfs1/gscratch/rao/vsathish/quals/saved_models/state_network/feb_13_run_3_embedding.params"

####################
# Load Model
####################

try:
    model_params = json.load(open(PARAMS_PATH, "r"))
    print("################## PARAMS LOADED #################")
    print(model_params)
    model = LearnableEmbedding(device, BATCH_SIZE, TIMESTEPS, model_params).to(device)

except:
    print("################## USING DEFAULT PARAMS #################")
    model = LearnableEmbedding(device, BATCH_SIZE, TIMESTEPS, INFER_PARAMS).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("######### DID NOT LOAD #########")
    sys.exit(0)


print("################## FREEZING MODEL PARAMS #######################")
for param in model.parameters():
    print(param.shape)
    param.requires_grad = False

#####################
# Load Data
#####################

data1, data2, data3, data4, data5 = generate_data(BATCH_SIZE, device, num_envs=NUM_ENVS, timesteps=TIMESTEPS)
train_x1, train_y1, test_x1, test_y1 = data1
train_x2, train_y2, test_x2, test_y2 = data2
train_x3, train_y3, test_x3, test_y3 = data3
train_x4, train_y4, test_x4, test_y4 = data4
train_x5, train_y5, test_x5, test_y5 = data5


print(train_x1[0].shape, train_y1[0].shape)
l1, _, higher1 = model(train_x1[0], train_y1[0], eval_mode=True) 
l2, _, higher2 = model(train_x2[0], train_y2[0], eval_mode=True) 
l3, _, higher3 = model(train_x3[0], train_y3[0], eval_mode=True) 
l4, _, higher4 = model(train_x4[0], train_y4[0], eval_mode=True) 
l5, _, higher5 = model(train_x5[0], train_y5[0], eval_mode=True) 

print(l1, l2, l3, l4, l5)

higher1 = np.array(higher1)
higher2 = np.array(higher2)
higher3 = np.array(higher3)
higher4 = np.array(higher4)
higher5 = np.array(higher5)


# %%
t1 = higher1[-1, :, :]
t2 = higher2[-1, :, :]
t3 = higher3[-1, :, :]
t4 = higher4[-1, :, :]
t5 = higher5[-1, :, :]

print(t1.shape, t2.shape, t3.shape, t4.shape, t5.shape)

m1 = np.mean(t1, axis=0)
m2 = np.mean(t2, axis=0)
m3 = np.mean(t3, axis=0)
m4 = np.mean(t4, axis=0)
m5 = np.mean(t5, axis=0)

print(m1, m2, m3)
print(np.square(m1-m2).sum())


#######################################
# TSNE for Episodic Convergence of S_T
#######################################

t1 = higher1[:, 0, :]
t2 = higher2[:, 0, :]
t3 = higher3[:, 0, :]
t4 = higher4[:, 0, :]
t5 = higher5[:, 0, :]

print(t1.shape, t2.shape, t3.shape)
print("First embeddings: ", t1[0], t2[0], t3[0])

episodes = np.concatenate([t1, t2, t3, t4, t5], axis=0)
# episodes = np.concatenate([t1, t2, t3, t4], axis=0)
e_tsne = TSNE(n_components=2, init='pca', perplexity=20)
e_embed = e_tsne.fit_transform(episodes)

print("########## CHECKING COMBINED TSNE SHAPE ##########")
print(episodes.shape, e_embed.shape)

size = len(e_embed)

# p1 = int(size/2)
p1 = int(size/5)
p2 = 2 * int(size/5)
p3 = 3 * int(size/5)
p4 = 4 * int(size/5)

x1 = e_embed[:p1, 0]
x2 = e_embed[p1:p2, 0]
x3 = e_embed[p2:p3, 0]
x4 = e_embed[p3:p4, 0]
x5 = e_embed[p4:, 0]

y1 = e_embed[:p1, 1]
y2 = e_embed[p1:p2, 1]
y3 = e_embed[p2:p3, 1]
y4 = e_embed[p3:p4, 1]
y5 = e_embed[p4:, 1]

print("First Embeddings: ", x1[0], y1[0], x2[0], y2[0], x3[0], y3[0])

plt.clf()
plt.close()

alpha0 = 1/len(x1)

_arr = [x for x in range(len(x1))]
colors1 = []
colors2 = []
colors3 = []
colors4 = []
colors5 = []

# print(_arr)
for x in _arr:
    alpha = alpha0 * (x+1)
    c1 = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, alpha)
    c2 = (1.0, 0.4980392156862745, 0.054901960784313725, alpha)
    c3 = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, alpha)
    c4 = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, alpha)
    c5 = (0.5803921568627451, 0.403921568627451, 0.7411764705882353, alpha)
    
    colors1.append(c1)
    colors2.append(c2)
    colors3.append(c3)
    colors4.append(c4)
    colors5.append(c5)


plt.scatter(x1, y1, color=colors1, label="Env 1")
plt.scatter(x2, y2, color=colors2, label="Env 2")
plt.scatter(x3, y3, color=colors3, label="Env 3")
plt.scatter(x4, y4, color=colors4, label="Env 4")
plt.scatter(x5, y5, color=colors5, label="Env 5")

plt.plot(x1, y1, color=c1, linewidth=0.2)
plt.plot(x2, y2, color=c2, linewidth=0.2)
plt.plot(x3, y3, color=c3, linewidth=0.2)
plt.plot(x4, y4, color=c4, linewidth=0.2)
plt.plot(x5, y5, color=c5, linewidth=0.2)

plt.scatter(x1[-1], y1[-1], marker='*', label="End", color="black", s=100)
plt.scatter(x2[-1], y2[-1], marker='*', color="black", s=100)
plt.scatter(x3[-1], y3[-1], marker='*', color="black", s=100)
plt.scatter(x4[-1], y4[-1], marker='*', color="black", s=100)
plt.scatter(x5[-1], y5[-1], marker='*', color="black", s=100)

plt.scatter(x1[0], y1[0], marker='D', color="black", s=65)
plt.scatter(x2[0], y2[0], marker='D', color="black", s=65)
plt.scatter(x3[0], y3[0], marker='D', color="black", s=65)
plt.scatter(x4[0], y4[0], marker='D', color="black", s=65)
plt.scatter(x5[0], y5[0], marker='D', color="black", s=65)

one = mlines.Line2D([], [], color=c1, marker='o', ls='', label='Environment 1')
two = mlines.Line2D([], [], color=c2, marker='o', ls='', label='Environment 2')
three = mlines.Line2D([], [], color=c3, marker='o', ls='', label='Environment 3')
four = mlines.Line2D([], [], color=c4, marker='o', ls='', label='Environment 4')
five = mlines.Line2D([], [], color=c5, marker='o', ls='', label='Environment 5')
six = mlines.Line2D([], [], color='black', marker='D', ls='', label='Episode Start')
seven = mlines.Line2D([], [], color='black', marker='*', ls='', label='Episode End')

plt.legend(handles=[one, two, three, four, five, six, seven])

# plt.legend()
plt.title("TSNE of latent converging over an episode - embedding method")
plt.savefig(PLOT_SAVE_PATH+"embedding_episodic_TSNE.png", dpi=300)
# sys.exit(0)

##################
# TSNE Embeddings
##################

h_concat = np.concatenate([higher1[-1], higher2[-1], \
                        higher3[-1], higher4[-1], higher5[-1]], axis=0)
print("######### EMBEDDINGS #########")
print(h_concat.shape, type(h_concat))

tsne_ = TSNE(n_components=2, init='pca', perplexity=50)
h_embed = tsne_.fit_transform(h_concat)
print(h_embed.shape)
# h_center1 = np.mean()
# h_center2 = h_embed[-1]
# h_center3 = h_embed[-3]
# h_embed = h_embed[:-3]

# print(h_center1, h_center2, h_center3)
print(h_embed.shape)

# sys.exit(0)

# Plotting
size = len(h_embed)
# p1 = int(size/2)
p1 = int(size/5)
p2 = 2 * int(size/5)
p3 = 3 * int(size/5)
p4 = 4 * int(size/5)


print(size)
x1 = h_embed[:p1, 0]
x2 = h_embed[p1:p2, 0]
x3 = h_embed[p2:p3, 0]
x4 = h_embed[p3:p4, 0]
x5 = h_embed[p4:, 0]

y1 = h_embed[:p1, 1]
y2 = h_embed[p1:p2, 1]
y3 = h_embed[p2:p3, 1]
y4 = h_embed[p3:p4, 1]
y5 = h_embed[p4:, 1]

h_center1 = [np.mean(x1), np.mean(y1)]
h_center2 = [np.mean(x2), np.mean(y2)]
h_center3 = [np.mean(x3), np.mean(y3)]
h_center4 = [np.mean(x4), np.mean(y4)]
h_center5 = [np.mean(x5), np.mean(y5)]

MU1 = [np.mean(x1), np.mean(y1)]
STD1 = [np.std(x1), np.std(y1)]
MU2 = [np.mean(x2), np.mean(y2)]
STD2 = [np.std(x2), np.std(y2)]

plt.clf()
plt.close()

# gauss2d(MU1, STD1, True)
# gauss2d(MU2, STD2, True)

plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.scatter(x3, y3, label='Env 3')
plt.scatter(x4, y4, label='Env 4')
plt.scatter(x5, y5, label='Env 5')
# plt.scatter(h_center1[0], h_center1[1], label="Env 1 center", marker='*', s=100)
# plt.scatter(h_center2[0], h_center2[1], label="Env 2 center", marker='*', s=100)
# plt.scatter(h_center3[0], h_center3[1], label="Env 3 center", marker='*', s=100)
# plt.scatter(h_center4[0], h_center4[1], label="Env 4 center", marker='*', s=100)
# plt.scatter(h_center5[0], h_center4[1], label="Env 5 center", marker='*', s=100)

# plt.scatter(x4, y4, label="Env 4")
plt.legend()
plt.title("TSNE of higher latents - embedding method")
plt.savefig(PLOT_SAVE_PATH+"embedding_state_cloud_TSNE.png")


# %%
##################
# PCA Embeddings
##################

h_concat = np.concatenate([higher1[-1], higher2[-1], higher3[-1], \
                           higher4[-1], higher5[-1]], axis=0)
print("######### EMBEDDINGS #########")
print(h_concat.shape, type(h_concat))

pca_ = PCA(n_components=2)
pca_embed = pca_.fit_transform(h_concat)
# h_center1 = pca_embed[-2]
# h_center2 = pca_embed[-1]
# h_center3 = pca_embed[-3]
# pca_embed = pca_embed[:-2]

# print(h_center1, h_center2, h_center3)
print(h_embed.shape)



size = len(pca_embed)
# p1 = int(size/2)
p1 = int(size/5)
p2 = 2 * int(size/5)
p3 = 3 * int(size/5)
p4 = 4 * int(size/5)


x1 = pca_embed[:p1, 0]
x2 = pca_embed[p1:p2, 0]
x3 = pca_embed[p2:p3, 0]
x4 = pca_embed[p3:p4, 0]
x5 = pca_embed[p4:, 0]

y1 = pca_embed[:p1, 1]
y2 = pca_embed[p1:p2, 1]
y3 = pca_embed[p2:p3, 1]
y4 = pca_embed[p3:p4, 1]
y5 = pca_embed[p4:, 1]

MU1 = [np.mean(x1), np.mean(y1)]
STD1 = [np.std(x1), np.std(y1)]
MU2 = [np.mean(x2), np.mean(y2)]
STD2 = [np.std(x2), np.std(y2)]

plt.clf()
plt.close()

# gauss2d(MU1, STD1, True)
# gauss2d(MU2, STD2, True)

h_center1 = [np.mean(x1), np.mean(y1)]
h_center2 = [np.mean(x2), np.mean(y2)]
h_center3 = [np.mean(x3), np.mean(y3)]
h_center4 = [np.mean(x4), np.mean(y4)]
h_center5 = [np.mean(x5), np.mean(y5)]


plt.scatter(x1, y1, label="Env 1")
plt.scatter(x2, y2, label="Env 2")
plt.scatter(x3, y3, label='Env 3')
plt.scatter(x4, y4, label='Env 4')
plt.scatter(x5, y5, label='Env 5')
# plt.scatter(h_center1[0], h_center1[1], label="Env 1 center", marker='*', s=100)
# plt.scatter(h_center2[0], h_center2[1], label="Env 2 center", marker='*', s=100)
# plt.scatter(h_center3[0], h_center3[1], label="Env 3 center", marker='*', s=100)
# plt.scatter(h_center4[0], h_center4[1], label="Env 4 center", marker='*', s=100)
# plt.scatter(h_center5[0], h_center4[1], label="Env 5 center", marker='*', s=100)


plt.legend()
plt.title("PCA of higher latents - embedding method")
plt.savefig(PLOT_SAVE_PATH+"embedding_cloud_PCA.png")
