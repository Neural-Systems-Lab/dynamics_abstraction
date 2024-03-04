import sys
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage



class Reconstructions:
    def __init__(self, model, device, batch_size):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.data_in_dims = 13
        self.action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.hypernet = self.model.hypernet
        self.temporal = self.model.temporal
        

    def predict_states(self, higher_state):
        
        PLOT_TIMESTEPS = 500
        PLOT_BATCH_SIZE = 100
        FPS = 10
        canvas = np.ones((5, 5)) # Initialize the cells with walls
        current_state = (2, 2) # Start from the center
        '''
        Hidden :  torch.Size([1, 100, 32])
        Weights :  torch.Size([100, 32])
        Batched Inputs torch.Size([100, 13])
        '''

        print("center : ", higher_state.shape)
        higher_state_batched = higher_state.repeat((PLOT_BATCH_SIZE, 1))
        weights = self.hypernet(higher_state_batched)
        print("weights : ", weights.shape)


        # Use a batch of random initial states and same actions and average out the predicted states
        # Generate for each timestep


        hidden = torch.zeros((self.temporal.num_layers, self.batch_size, \
                self.temporal.hidden_size), device=self.device)
        print("hidden : ", hidden.shape)

        batched_init_state = torch.randint(0, 2, (PLOT_BATCH_SIZE, self.data_in_dims-4)).float()
        random_action = F.one_hot(torch.randint(0, 4, (1,)), num_classes=4).repeat(PLOT_BATCH_SIZE, 1)
        batch_input = torch.cat([batched_init_state, random_action], dim=1).float().to(self.device)

        print("Batched Inputs", batch_input.shape)
        
        predicted_state_list = [torch.mean(batched_init_state, axis=0).view(3, 3).cpu().numpy()]
        action_list = [np.argmax(random_action[0].cpu().numpy())]
        patch_pos_list = [current_state]
        np.set_printoptions(precision=3) 
        # Get the empty frame
        # cur_frame, canvas = self.plot_predictions(canvas, current_state, predicted_state_list[0], 0)
        env_frames = []
        print("Initial canvas")
        print(canvas)
        for timestep in range(PLOT_TIMESTEPS):
            
            print("Timestep : ", timestep)

            # Predict the next state
            hidden = hidden.detach()
            predicted_states, hidden = self.temporal.forward_inference(batch_input, weights, hidden)
            predicted_ = torch.round(torch.mean(predicted_states, axis=0))
            patch = predicted_.view(3, 3)
            print("Predicted states : ", predicted_.shape)
            print(patch)

            # Get the next action and update canvas state
            # actions = self.get_possible_actions(canvas, current_state)
            actions = [0, 1, 2, 3]
            action_ = np.random.choice(actions)
            current_state = self.take_step(current_state, action_, canvas)

            if timestep % 20 == 0:
                current_state = (2, 2)
                hidden = torch.zeros((self.temporal.num_layers, self.batch_size, \
                            self.temporal.hidden_size), device=self.device)

            # Plot and get the frame
            cur_frame, canvas = self.plot_predictions(canvas, current_state, patch.cpu().numpy(), timestep)

            # Update the patch data
            env_frames.append(cur_frame)
            action_list.append(action_)
            predicted_state_list.append(patch.cpu().numpy())
            patch_pos_list.append(current_state)


            # Prepare the data for next step
            # Add some small noise to the predicted state
            action_one_hot = F.one_hot(torch.tensor(action_), num_classes=4).float().to(self.device)
            action_one_hot = action_one_hot.repeat(PLOT_BATCH_SIZE, 1)
            predicted_ = predicted_ + (0.5-torch.randn_like(predicted_))*0.25
            predicted_ = predicted_.repeat((PLOT_BATCH_SIZE, 1))
            
            batch_input = torch.cat([action_one_hot, predicted_], dim=1)
            print("Batched Inputs", batch_input.shape)
            self.model.zero_grad()
        
        # Generate the video
        # clip = VideoClip(lambda t: env_frames[int(t)], duration=PLOT_TIMESTEPS)
        clip = ImageSequenceClip(env_frames, fps=15)
        clip.write_videofile("/mmfs1/gscratch/rao/vsathish/quals/plots/patch_predictions/"+"patch_predictions.mp4")
        # clip.write_gif("/mmfs1/gscratch/rao/vsathish/quals/plots/patch_predictions/"+"patch_predictions.gif", fps=20)
        

    def plot_predictions(self, canvas, cur_state, predictions, t):
        
        # Updated canvas
        alpha = 0.9
        alpha2 = 0.9
        x, y = cur_state[0]-1, cur_state[1]-1
        # canvas[cur_state] = 0
        for i in range(3):
            for j in range(3):
                if x+i >= 0 and x+i < 5 and y+j >= 0 and y+j < 5:
                    if predictions[i, j] < 0.5:
                        canvas[x+i, y+j] = alpha2*canvas[x+i, y+j] + (1-alpha2)*predictions[i, j]
                    else:
                        canvas[x+i, y+j] = alpha*canvas[x+i, y+j] + (1-alpha)*predictions[i, j]
        
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
        x, y = cur_state[0]-1, cur_state[1]-1
        rect = patches.Rectangle((y-0.5, x-0.5), 3, 3, linewidth=2, edgecolor='r', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        # ax.imshow(canvas, cmap=cmap, norm=norm)
        # plt.savefig("/mmfs1/gscratch/rao/vsathish/quals/plots/patch_predictions/"+str(t)+".png")

        return mplfig_to_npimage(fig), canvas


    def get_pomdp_state(self, canvas, current_state):
        # Extract a 3 x 3 patch around the current state
        # Return a flattened version of the patch
        # print(self.state)
        patch = np.zeros((3, 3))
        for i in range(0, 3):
            for j in range(0, 3):
                r, c = current_state[0]-1+i, current_state[1]-1+j
                if r < 0 or r >= self.rows or c < 0 or c >= self.columns:
                    patch[i][j] = np.random.choice((0, 1))
                else:
                    patch[i][j] = canvas[(r, c)]
        return patch.flatten()

    def get_possible_actions(self, canvas, current_state):

        possible_actions = []
        next_states = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i, action in enumerate(next_states):
            x, y = current_state[0]+action[0], current_state[1]+action[1]
            if x < 0 or x > 4 or y < 0 or y > 4:
                continue
            if canvas[x,y] < 0.5:
                possible_actions.append(i)
        
        print("Possible actions : ", [self.action_map[i] for i in possible_actions], possible_actions)
        
        if len(possible_actions) == 0:
            print("No possible actions, returning random action")
            return [np.random.choice([0, 1, 2, 3])]
        return possible_actions
    
    def take_step(self, current_state, action, canvas):
        print("##### TAKING a step ... ######")
        print("Current state : ", current_state)
        print("Action : ", action)
        print("Canvas : ")
        print(canvas)

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
            return current_state
        elif canvas[next_state] > 0.5:
            print("Hitting a wall, returning same state ...")
            # if np.random.uniform() > 0.95:
            #     return next_state
            return current_state
        else:
            print("Next state : ", next_state)
            return next_state
