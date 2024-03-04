# %%
import os
import sys
import json
sys.path.insert(1, "/gscratch/rao/vsathish/quals/dynamics_abstraction")
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits import mplot3d
from copy import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal


# from models.embedding_model import LearnableEmbedding, INFER_PARAMS
from models.embedding_ce import LearnableEmbedding, MODEL_PARAMS
from dataloaders.dataloader_pomdp import generate_data, configs
from viz.token_reconstruct import Reconstructions
from environments.env import SimpleGridEnvironment
from environments.composition import CompositionGrid
from environments.pomdp_config import *

device = torch.device("cuda")
BATCH_SIZE = 100
# SAMPLES = 600
NUM_ENVS = 2
N_COMPONENTS = 2
TIMESTEPS = 5

date_ = "mar2"
run_ = 1010
S_dims = 32

ROOT = "/mmfs1/gscratch/rao/vsathish/quals/saved_models/state_network/"
MODEL_PATH = ROOT+date_+"_run_"+str(run_)+"_dims_"+str(S_dims)+"_envs_"+str(NUM_ENVS)+"_embedding.state"
PARAMS_PATH = ROOT+date_+"_run_"+str(run_)+"_dims_"+str(S_dims)+"_envs_"+str(NUM_ENVS)+"_embedding.json"
PLOT_SAVE_PATH = "/mmfs1/gscratch/rao/vsathish/quals/plots/state_network/"

# env = SimpleGridEnvironment(config=c1)

color_pallette = [
    [0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1],
    [1.0, 0.4980392156862745, 0.054901960784313725, 1],
    [0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1],
    [0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1],
    [0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1],
    [0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1],
    [0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1],
    [0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1],
    [0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1],
    [0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1]

]

def get_viz_data(model):
    data = generate_data(BATCH_SIZE, device, num_envs=NUM_ENVS, timesteps=TIMESTEPS)
    higher_list = []
    assert len(data) == NUM_ENVS

    for i, data_ in enumerate(data):
        train_x, train_y, _, _ = data_
        loss, _, higher = model(train_x[0], train_y[0], eval_mode=True)
        print(f"#### Loss for env {i} : #####", loss.item())
        # print(f"#### Higher for env {i} : #####", higher[0].shape, train_x[0].shape, train_y[0].shape)
        higher_list.append(np.array(higher))
    
    return higher_list

def get_state_centers(higher_list):
    centers = []
    for i in range(NUM_ENVS):
        print("higher state shape: ", higher_list[i].shape)
        last_state = higher_list[i][-1]
        print("Last state shape: ", last_state.shape)
        mu = np.mean(last_state, axis=0)
        centers.append(mu.tolist())
    return centers

def get_episodes(higher_states):
    assert len(higher_states) == NUM_ENVS
    episodes = [] 
    for h in higher_states:
        episodes.append(h[:, 0, :])
    
    episodes = np.concatenate(episodes)
    return episodes

def plot_episodic_tsne(higher_states, episodic_embedding=None, save=''):
    # Randomly sample episodes from each environment and plot
    # episodes = [] 
    # for h in higher_states:
    #     episodes.append(h[:, 0, :])

    # assert len(higher_states) == NUM_ENVS
    
    episodes = get_episodes(higher_states)
    episodic_tsne = TSNE(n_components=N_COMPONENTS, perplexity=6)
    if episodic_embedding is None:
        print("Embedding episodic data from scratch")
        episodic_embedding = episodic_tsne.fit_transform(episodes)

    print("########## CHECKING COMBINED TSNE SHAPE ##########")
    print(episodes.shape, episodic_embedding.shape)

    size = len(episodic_embedding)
    split = [i*int(size/NUM_ENVS) for i in range(NUM_ENVS+1)]
    x, y = [], []
    if NUM_ENVS == 3:
        x, y, z = [], [], []

    for i in range(NUM_ENVS):
        m, n = split[i], split[i+1]
        x.append(episodic_embedding[m:n, 0])
        y.append(episodic_embedding[m:n, 1])
        if N_COMPONENTS == 3:
            z.append(episodic_embedding[m:n, 2])
    
    colors = [[] for _ in range(NUM_ENVS)]
    alpha0 = 1/(TIMESTEPS)

    for i in range(TIMESTEPS+1):
        alpha = alpha0 * i
        for j in range(NUM_ENVS):
            c = copy(color_pallette[j])
            c[3] = alpha
            colors[j].append(tuple(c))
    
    # Plot
    if N_COMPONENTS == 3:
        plot_3d(x, y, z)
        return episodes
    
    legend_handles = []
    for i in range(NUM_ENVS):
        # Plot points
        plt.scatter(x[i], y[i], color=colors[i], label="Env "+str(i+1))
        # Plot lines
        plt.plot(x[i], y[i], color=colors[i][-1], linewidth=1)
        # Plot the last step
        plt.scatter(x[i][-1], y[i][-1], marker='*', color=colors[i][-1], s=200)
        # Plot the first step
        plt.scatter(x[i][0], y[i][0], marker='D', color="black", s=65)
        # Plot legend handles
        legend_handles.append(mlines.Line2D([], [], color=colors[i][-1], marker='o', ls='', label='Environment '+str(i+1)))
    
    legend_handles.append(mlines.Line2D([], [], color='black', marker='D', ls='', label='Episode Start'))
    legend_handles.append(mlines.Line2D([], [], color='black', marker='*', ls='', label='Episode End'))

    plt.legend(handles=legend_handles)
    plt.title("TSNE of latent converging over an episode - embedding method")
    plt.savefig(PLOT_SAVE_PATH+"embedding_episodic_TSNE_"+save+".png", dpi=300)
    print("plot_success")
    return episodes

def plot_state_cloud(higher_states, tsne=False, episodes=[], interpolate=[], envs=NUM_ENVS, save=''):
    # If episodic is given, we plot that with the cloud
    assert len(higher_states) == envs

    higher_points = []
    print(len(higher_states), higher_states[0].shape)
    
    for i in range(envs):
        higher_points.append(higher_states[i][-1])
    
    if len(episodes) != 0:
        higher_points.append(episodes)

    # if len(interpolate) != 0:
    #     interpolate_arr = []
    #     alphas = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #     c1, c2, = interpolate[0], interpolate[1]
    #     for a in alphas:
    #         interpolate_arr.append(a * np.array(c1) + (1-a) * np.array(c2))
    #     print(interpolate_arr[0].shape)
    #     higher_points.append(np.array(interpolate_arr))

    h_concat = np.concatenate(higher_points, axis=0)
    if tsne:
        print("TSNE")
        tsne_ = TSNE(n_components=N_COMPONENTS, perplexity=60)
        h_embed = tsne_.fit_transform(h_concat)
    else:
        print("PCA")
        pca_ = PCA(n_components=N_COMPONENTS)
        h_embed = pca_.fit_transform(h_concat)
    
    print("########## COMBINED TSNE ##########")
    print(h_concat.shape, h_embed.shape)
    
    plt.clf()
    plt.close()

    # if len(interpolate) != 0:
    #     interpolated = 

    if len(episodes) != 0:
        episodes_embed = h_embed[-episodes.shape[0]:]
        h_embed = h_embed[:-episodes.shape[0]]
        print(h_embed.shape, episodes_embed.shape)
        # plot_episodic_tsne(higher_states, episodic_embedding=episodes_embed, save=save)
    
    size = len(h_embed)
    split = [i*int(size/envs) for i in range(envs+1)]
    x, y = [], []
    if envs == 3:
        x, y, z = [], [], []

    for i in range(envs):
        m, n = split[i], split[i+1]
        x.append(h_embed[m:n, 0])
        y.append(h_embed[m:n, 1])
        if N_COMPONENTS == 3:
            z.append(h_embed[m:n, 2])
    
    # Plot
    if N_COMPONENTS == 3:
        plot_3d(x, y, z)
        return

    for i in range(envs):
        if len(episodes) != 0:
            plt.scatter(x[i], y[i], label="Env "+str(i+1), alpha=0.3)
        else:
            plt.scatter(x[i], y[i], label="Env "+str(i+1), alpha=0.5)
            plt.legend()

    if len(episodes) != 0:
        print("Plotting Episodic Data")
        plot_episodic_tsne(higher_states, episodic_embedding=episodes_embed, save=save)

    if tsne:
        plt.title("TSNE of higher latents - embedding method")
        plt.savefig(PLOT_SAVE_PATH+"embedding_cloud_TSNE_"+save+".png", dpi=300)
    else:
        plt.title("PCA of higher latents - Embedding method")
        plt.savefig(PLOT_SAVE_PATH+"embedding_cloud_PCA_"+save+".png", dpi=300)   
    plt.clf()
    plt.close()
    
    return

def plot_3d(x, y, z):
    pass

def gauss2d(mu, sigma, to_plot=False):
    w, h = 50, 50

    std = [np.sqrt(sigma[0]), np.sqrt(sigma[1])]
    x = np.linspace(mu[0] - 0.5 * std[0], mu[0] + 0.5 * std[0], w)
    y = np.linspace(mu[1] - 0.5 * std[1], mu[1] + 0.5 * std[1], h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order='F')

    if to_plot:
        plt.contourf(x, y, z.T, levels=1, alpha=0.5)
        # plt.contourf(x, y, z.T)
        # plt.savefig("gauss2d.png", dpi=400)
        # plt.show()
    return z


# MU = [50, 70]
# SIGMA = [75.0, 90.0]
# z = gauss2d(MU, SIGMA, True)


if __name__=="__main__":
    print("Running Viz Embeddings")
    ####################################################
    # Load Model and Switch off gradients for inference
    ####################################################

    try:
        model_params = json.load(open(PARAMS_PATH, "r"))
        print("################## PARAMS LOADED #################")
        # print(model_params)
        model = LearnableEmbedding(device, BATCH_SIZE, TIMESTEPS, model_params).to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded Model")

    except:
        print("################## USING DEFAULT PARAMS #################")
        model = LearnableEmbedding(device, BATCH_SIZE, TIMESTEPS, MODEL_PARAMS).to(device)
        model.load_state_dict(torch.load(MODEL_PATH))

    # Switch off gradients
    for param in model.parameters():
        print(param.shape)
        param.requires_grad = False
    
    FILE_SUFFIX = 'mar_2_2'


    # Get data for plotting
    higher_states = get_viz_data(model)
    centers = get_state_centers(higher_states)

    if 'centers' not in model_params:
        model_params['centers'] = centers
        with open(PARAMS_PATH, "w") as f:
            json.dump(model_params, f, indent=4)
    
    # Plot the data
    episodes = plot_episodic_tsne(higher_states, save=FILE_SUFFIX)
    episodes = get_episodes(higher_states)
    plot_state_cloud(higher_states, tsne=False, episodes=[], envs=NUM_ENVS, save=FILE_SUFFIX)

    # Generate video of predictions
    recon = Reconstructions(model, device, BATCH_SIZE)
    # centers = model_params['centers']

    alpha = 0.38
    intepolation = alpha * np.array(centers[0]) + (1-alpha) * np.array(centers[1])
    center_ = torch.from_numpy(np.array(intepolation)).float().to(device)
    
    # recon.predict_states(center_) 