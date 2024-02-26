
import os
import sys
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F

from dataloaders.dataloader_pomdp import generate_data, configs
from dataloaders.dataloader_pomdp import TRAIN, TEST
from models.embedding_model import LearnableEmbedding, MODEL_PARAMS
from viz.utils import plot_state_cloud

###################
# CONSTANTS
###################

device = torch.device("cuda")
# device = torch.device("cpu")
HYPER_EPOCHS = 100
BATCH_SIZE = 100
TIMESTEPS = 10
NUM_ENVS = 3
TRAIN_BATCHES = TRAIN // BATCH_SIZE
TEST_BATCHES = TEST // BATCH_SIZE

'''
Change filenames during each run
'''
date_ = "feb23"
run_ = 6
S_dims = 32

ROOT = "/mmfs1/gscratch/rao/vsathish/quals/saved_models/state_network/"
MODEL_PATH = ROOT+date_+"_run_"+str(run_)+"_dims_"+str(S_dims)+"_envs_"+str(NUM_ENVS)+"_embedding.state"
PARAMS_PATH = ROOT+date_+"_run_"+str(run_)+"_dims_"+str(S_dims)+"_envs_"+str(NUM_ENVS)+"_embedding.json"

print(MODEL_PARAMS)
print(MODEL_PATH)
print(PARAMS_PATH)
#########################################
# Training a Hypernet Modulated Network
#########################################

# Get the data

# data1, data2 = generate_data(BATCH_SIZE, device)
# data1, data2, data3, data4, data5 = generate_data(BATCH_SIZE, device, \
#                                     num_envs=NUM_ENVS, timesteps=TIMESTEPS)
data1, data2, data3 = generate_data(BATCH_SIZE, device, \
                                    num_envs=NUM_ENVS, timesteps=TIMESTEPS)
train_x1, train_y1, test_x1, test_y1 = data1
train_x2, train_y2, test_x2, test_y2 = data2
train_x3, train_y3, test_x3, test_y3 = data3
# train_x4, train_y4, test_x4, test_y4 = data4
# train_x5, train_y5, test_x5, test_y5 = data5

############################
# MODEL AND OPTIMIZER LOAD
############################

try:
    model_params = json.load(open(PARAMS_PATH, "r"))
    print("################## PARAMS LOADED #################")
    print(model_params)
    model = LearnableEmbedding(device, BATCH_SIZE, TIMESTEPS, model_params).to(device)

except:
    print("################## USING DEFAULT PARAMS #################")
    model_params = MODEL_PARAMS
    model = LearnableEmbedding(device, BATCH_SIZE, TIMESTEPS, model_params).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("################## LOAD SUCCESS #################")

except:
    print("################## TRAINING FROM SCRATCH #######################")

hyper_optim = torch.optim.Adam(model.hypernet.parameters(), model.hyper_lr)
temporal_optim = torch.optim.Adam(model.temporal.parameters(), model.temporal_lr)

print("Everything cool till now")
train_loss = []
test_loss = []
center_distances = []

############################
# TRAINING
############################

for epochs in range(HYPER_EPOCHS):
    train_loss_ = []
    test_loss_ = []
    center_distances_ = []
    print("########## Epoch : ", epochs, "##########")

    for i in range(TRAIN_BATCHES): # batches = num_trajectories / batch_size
        
        # Ignore predicted and inferred states during training
        hyper_optim.zero_grad()
        temporal_optim.zero_grad()
    
        l1, centers1 = model(train_x1[i], train_y1[i]) 
        l2, centers2 = model(train_x2[i], train_y2[i])
        l3, centers3 = model(train_x3[i], train_y3[i])
        # l4, centers4 = model(train_x4[i], train_y4[i])
        # l5, centers5 = model(train_x5[i], train_y5[i])

        center_magnitudes = sum([torch.norm(c) for c in [centers1, centers2]])
        loss = l1+l2+l3
        # loss = l1+l2+l3+l4+l5 #- center_magnitudes
        loss.backward()
        hyper_optim.step()
        temporal_optim.step()
        hyper_optim.zero_grad()
        temporal_optim.zero_grad()

        print("i = ", i, "loss for Env 1, 2, ... = ", l1.detach().cpu().numpy(), \
                l2.detach().cpu().numpy(), l3.detach().cpu().numpy(),\
                # l4.detach().cpu().numpy(), l5.detach().cpu().numpy(), \
              "\ncenter_magnitudes = ", center_magnitudes.detach().cpu().numpy())
        
        centers = [x.detach().cpu().numpy() for x in [centers1, centers2, centers3]]
        train_loss_.append(loss.detach().cpu().numpy())
        centers_dist_arr = []

        for i, j in list(itertools.combinations(range(len(centers)), 2)):
            dist = np.linalg.norm(centers[i-1]-centers[j-1])
            centers_dist_arr.append(dist)
        
        center_distances_.append(sum(centers_dist_arr)/len(centers_dist_arr))

    for i in range(TEST_BATCHES):
        l1, _, higher1 = model(test_x1[i], test_y1[i], eval_mode=True) 
        l2, _, higher2 = model(test_x2[i], test_y2[i], eval_mode=True)
        l3, _, higher3 = model(test_x3[i], test_y3[i], eval_mode=True)
        # l4, _, higher4 = model(test_x4[i], test_y4[i], eval_mode=True)
        # l5, _, higher5 = model(test_x5[i], test_y5[i], eval_mode=True)
        # loss = l1+l2+l3+l4
        loss = l1+l2+l3#+l4+l5
        test_loss_.append(loss)

        if epochs % 10 == 0 and i == 0:
            print("Plotting PCA cloud ...")
            higher_list = np.array([higher1, higher2, higher3])
            centers = plot_state_cloud(higher_list, tsne=False, envs=NUM_ENVS, save="feb_23_3")



    print("Mean Train Loss (Per Step): ", np.mean(train_loss_)/(NUM_ENVS*TIMESTEPS))
    train_loss.append(np.mean(train_loss_)/(NUM_ENVS*TIMESTEPS))
    print("Mean Test Loss (Per Step): ", np.mean(test_loss_)/(NUM_ENVS*TIMESTEPS))
    test_loss.append(np.mean(test_loss_)/(NUM_ENVS*TIMESTEPS))
    print("Mean Center Distance: ", np.mean(center_distances_))
    center_distances.append(np.mean(center_distances_))

    if (epochs+1) % 10 == 0 and epochs != 0:
        print("Saving Checkpoint ... ")

        model_params["train_loss"] = train_loss
        model_params["test_loss"] = test_loss
        model_params["center_distances"] = center_distances

        torch.save(model.state_dict(), MODEL_PATH)
        json.dump(model_params, open(PARAMS_PATH, "w"), indent=4)
        
        plt.clf()
        plt.close()
        plt.plot(train_loss, label="Train Loss")
        plt.plot(test_loss, label="Val Loss")
        # plt.plot(center_distances, label="Center Distances")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Embedding State Model Loss")
        plt.savefig("../plots/loss/embedding_loss"+date_+"_dims_"+str(S_dims)+".png")
        
torch.save(model.state_dict(), MODEL_PATH)