import sys
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.set_printoptions(precision=3)

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage

from environments.pomdp_config import *
from environments.env import SimpleGridEnvironment


class Reconstructions:
    def __init__(self, model, device, batch_size):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.data_in_dims = 13
        self.action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        # self.action_map_old = {0: "RIGHT", 1: "LEFT", 2: "DOWN", 3: "UP"}
        self.hypernet = self.model.hypernet
        self.temporal = self.model.temporal
        self.env = SimpleGridEnvironment(config=c1) # To access mapping functions

    def predict_states(self, higher_state):
        
        PLOT_TIMESTEPS = 200
        PLOT_BATCH_SIZE = 100
        FPS = 10
        canvas = np.ones((5, 5)) # Initialize the cells with walls
        current_state = (2, 2) # Start from the center
        canvas[current_state] = 0
        start_one_hot = self.env.one_hot_mapping[(2, 2)]

        higher_state_batched = higher_state.repeat((PLOT_BATCH_SIZE, 1))
        weights = self.hypernet(higher_state_batched)

        hidden = torch.zeros((self.temporal.num_layers, self.batch_size, \
                self.temporal.hidden_size), device=self.device)

        batched_init_state = torch.from_numpy(start_one_hot).repeat(PLOT_BATCH_SIZE, 1).float()
        random_action = F.one_hot(torch.randint(0, 4, (1,)), num_classes=4).repeat(PLOT_BATCH_SIZE, 1)
        batch_input = torch.cat([batched_init_state, random_action], dim=1).float().to(self.device)

        print("Batched Inputs", batch_input.shape, batch_input[0], batch_input[1])
        
        predicted_state_list = [start_one_hot]
        action_ = np.argmax(random_action[0].cpu().numpy())
        print("Action : ", self.action_map[action_])
        action_list = []
        patch_pos_list = [current_state]

        env_frames = []
        print("Initial canvas")
        print(canvas)

        for timestep in range(PLOT_TIMESTEPS):
            print("Timestep : ", timestep)

            # Predict the next states
            hidden = hidden.detach()
            predicted_states, hidden = self.temporal.forward_inference(batch_input, weights, hidden)
            predicted_ = torch.mean(predicted_states, axis=0)
            # print("Predicted State : ", predicted_.detach().cpu().numpy())
            pred_state = self.env.inverted_one_hot[int(np.argmax(predicted_.detach().cpu().numpy()))]
            # print("Predicted State : ", pred_state)

            # Update the current state
            canvas_state, current_state = self.take_step(current_state, action_, pred_state, canvas)

            # Plot and get the frame. Changes the canvas
            cur_frame, canvas = self.plot_predictions(canvas, canvas_state, current_state, timestep)
            env_frames.append(cur_frame)

            if timestep % 15 == 0:
                current_state = (2, 2) # Reset to center
                hidden = torch.zeros((self.temporal.num_layers, self.batch_size, \
                                self.temporal.hidden_size), device=self.device)
            # Prepare the data for next step
            # Add some small noise to the predicted state
            actions = [0, 1, 2, 3]
            action_ = np.random.choice(actions)
            action_one_hot = F.one_hot(torch.tensor(action_), num_classes=4).float()
            random_action = action_one_hot.repeat(PLOT_BATCH_SIZE, 1)
            
            state_one_hot = self.env.one_hot_mapping[current_state]
            batched_init_state = torch.from_numpy(state_one_hot).repeat(PLOT_BATCH_SIZE, 1).float()
            batch_input = torch.cat([batched_init_state, random_action], dim=1).float().to(self.device)

        clip = ImageSequenceClip(env_frames, fps=15)
        clip.write_videofile("/mmfs1/gscratch/rao/vsathish/quals/plots/patch_predictions/"+"patch_predictions.mp4")

    def take_step(self, current_state, action, predicted_state, canvas):
        print("##### TAKING a step ... ######")
        print("Previous state : ", current_state)
        print("Action : ", self.action_map[action])
        print("Predicted state : ", predicted_state)
        
        next_state = current_state
        if action == 0:
            next_state = (current_state[0]-1, current_state[1])
        elif action == 1:
            next_state = (current_state[0]+1, current_state[1])
        elif action == 2:
            next_state = (current_state[0], current_state[1]-1)
        elif action == 3:
            next_state = (current_state[0], current_state[1]+1)
        
        if next_state[0] < 0 or next_state[0] > 4 or next_state[1] < 0 or next_state[1] > 4:
            print("Crossing borders, returning same state ...")
            return current_state, current_state
        # elif canvas[next_state] > 0.5:
        #     print("Hitting a wall, returning same state ...")
        #     # if np.random.uniform() > 0.95:
        #     #     return next_state
        #     return current_state
        elif next_state == predicted_state:
            print("Predicted state is same as estimated state")
            return next_state, next_state
    
        elif predicted_state == (0, 0):
            print("Predicted state is invalid")
            return next_state, (2, 2)
        else:
            print("Next state not the same as estimate. Not moving ", next_state, predicted_state)
            return next_state, predicted_state


    def plot_predictions(self, canvas, canvas_state, predicted_state, t):
        
        # Updated canvas
        alpha = 0.9
        if canvas_state == predicted_state:
            canvas[canvas_state] = 0
        else:
            canvas[canvas_state] = alpha * canvas[canvas_state] + (1-alpha) * 1
        
        print(canvas)

        # Now plot
        plt.clf()
        plt.close()

        # cmap = colors.ListedColormap(['black', 'gray'])
        # bounds = [0, 0.6]
        # norm = colors.BoundaryNorm(bounds, cmap.N)
        
        fig, ax = plt.subplots()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(which='major', axis='both', linestyle='-', color='darkgray', linewidth=1)
        ax.set_xticks(np.arange(-0.5, 100.5, 1))
        ax.set_yticks(np.arange(-0.5, 100.5, 1))

        plt.title("Drawing the Environment with higher states")
        ax.imshow(canvas, cmap='gray', vmin=0, vmax=2)
        _x, _y = predicted_state[0]-1, predicted_state[1]-1
        rect = patches.Rectangle((_y-0.5, _x-0.5), 3, 3, linewidth=2, edgecolor='r', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        # ax.imshow(canvas, cmap=cmap, norm=norm)
        # plt.savefig("/mmfs1/gscratch/rao/vsathish/quals/plots/patch_predictions/"+str(t)+".png")

        return mplfig_to_npimage(fig), canvas