import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys

import functools
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage


###########################################
# Network with learnable Stories
###########################################

MODEL_PARAMS = {
    "input_units": 32,
    "output_units": 16,
    "data_in_dims": 13,
    "data_out_dims": 9,
    "input_timesteps": 20,
    "infer_lr": 0.1,
    "hyper_lr": 0.001,
    "temporal_lr": 0.001,
    "infer_max_iter": 20,
    "l2_lambda": 0.0001,
    "var_lambda": 0.01,
    # "center_lambda": 0.001,
    "hypernet_layers": [128, 256, 256],

}

INFER_PARAMS = {
    "input_units": 32,
    "output_units": 16,
    "data_in_dims": 13,
    "data_out_dims": 9,
    "input_timesteps": 25,
    "infer_lr": 0.1,
    "hyper_lr": 0.0005,
    "temporal_lr": 0.0005,
    "infer_max_iter": 20,
    "l2_lambda": 0.001,
    "var_lambda": 0.0001,
    "hypernet_layers": [128, 256, 256],

}

class LearnableEmbedding(nn.Module):
    def __init__(self, device, batch_size, timesteps=25, params=MODEL_PARAMS):
        super(LearnableEmbedding, self).__init__()
        
        self.input_units = params["input_units"]
        self.output_units = params["output_units"]
        self.data_in_dims = params["data_in_dims"]
        self.data_out_dims = params["data_out_dims"]
        self.input_timesteps = timesteps
        self.device = device
        self.batch_size = batch_size

        # Learning rates
        self.infer_lr = params["infer_lr"]
        self.hyper_lr = params["hyper_lr"]
        self.temporal_lr = params["temporal_lr"]
        
        # Other vars
        self.infer_max_iter = params["infer_max_iter"]
        self.l2_lambda = params["l2_lambda"]
        if "var_lambda" in params:
            self.var_lambda = params["var_lambda"]
        else:
            self.var_lambda = 0.001

        # Models
        layers = params["hypernet_layers"]
        assert len(layers) > 2

        self.hypernet = nn.Sequential(
            nn.Linear(self.input_units, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], self.output_units)
        )
        self.temporal = LowerRNN(self.output_units, device, \
                        batch_size, self.data_in_dims, self.data_out_dims)
        
        self.action_map = {
                0:"up",
                1:"down",
                2:"left",
                3:"right"
            }

    ####################
    # Batched functions
    ####################
    
    def forward(self, temporal_batch_input, temporal_batch_output, eval_mode=False):
    
        # errors = torch.zeros(1, requires_grad=True).to(self.device)
        errors = 0
        # errors = []
        predicted_states_list = []
        higher_state_list = []
        weights_list = []
        higher_state = None
        
        if eval_mode:
            self.eval_mode = True
        else:
            self.eval_mode = False
        
        higher_state_list.append(np.squeeze(self.batch_init_story().clone().detach().cpu().numpy()))
        
        for t in range(len(temporal_batch_input)):
        # for state, action, next_state in trajectory:
            
            higher_state = self.batch_inference(temporal_batch_input[t], \
                            temporal_batch_output[t], t, higher_state)
            
            if eval_mode:
                higher_state_list.append(torch.squeeze(higher_state).detach().cpu().numpy())
        
        # Find the center of the higher state clusters for the last step
        cluster_centers = self.cluster_centers(higher_state)

        # This is the final weights for this batch of samples
        weights = self.hypernet(higher_state)
        # print("Weights : ", weights[0].shape, "Higher states : ", higher_state[0])
        
        # print("Higher states : ", higher_state[0])
        temporal_batched_weights = weights.repeat(self.input_timesteps, 1, 1)
        # print("shapes : ", temporal_batch_input.shape, temporal_batch_output.shape, weights.shape)

        predicted_states = self.temporal(temporal_batch_input, temporal_batched_weights)
        # print("###### predicted states : ", predicted_states[-1][0])
        # print("###### actual states : ", temporal_batch_output[-1][0])

        errors = torch.pow(predicted_states-temporal_batch_output, 2)
        # print(errors.shape)
        errors = torch.sum(torch.sum(errors, axis=-1), axis=0)
        # print(errors.shape)
        errors = torch.mean(errors)
        
        if eval_mode:
            return errors.detach().cpu().numpy(), \
                predicted_states_list, higher_state_list#, weights_list
                
        # return errors/len(temporal_batch_input)
        return errors, cluster_centers
    
    def batch_inference(self, batch_input, batch_output, timestep, previous_story=None):
        
        if previous_story == None:
            story = self.batch_init_story()
        else:
            story = previous_story.clone().detach()
            story.requires_grad = True
        
        # Is this optimizer definition correct for batched story?
        # infer_optimizer = torch.optim.SGD([story], self.infer_lr*(1/(timestep+1)))
        infer_optimizer = torch.optim.Adam([story], self.infer_lr)
        # infer_optimizer = torch.optim.SGD([story], self.infer_lr)
        
        hidden = torch.zeros((self.temporal.num_layers, self.batch_size, \
                self.temporal.hidden_size), device=self.device, requires_grad=True)
        
        for i in range(self.infer_max_iter):
            weights = self.hypernet(story)

            hidden = hidden.detach()
            predicted_states, hidden = self.temporal.forward_inference(batch_input, weights, hidden)
            # print("Temporal next states : ", predicted_states.shape)
            
            '''
            # Compute Loss.
            # Check if L2 norm gives better results
            # Check if variance/STD gives a better result
            '''
            loss = self.batch_errors(batch_output, predicted_states) +\
                    self.var_lambda * torch.sum(torch.var(story, axis=0)) +\
                    self.l2_lambda * torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)
                    
                # self.l1_lambda * torch.mean(torch.sum(torch.abs(story), axis=-1), axis=0)
            
            loss.backward()
            infer_optimizer.step()
            infer_optimizer.zero_grad()
            self.zero_grad()
            # print("backward pass done for step : ", i)
        
        if self.eval_mode and timestep % 20 == 0:
            print("Variance : ", torch.sum(torch.var(story, axis=0)).detach().cpu().numpy())
            print("L2 val : ", torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)\
                 .detach().cpu().numpy(), "\nLoss : ", loss.detach().cpu().numpy())
            
        return story.clone().detach()
        # return story.clone()

    def batch_errors(self, true, predicted):  
        err = torch.pow(true-predicted, 2)
        err = torch.sum(err, axis=-1)
        err = torch.mean(err, axis=0)
        return err
    
    def cluster_centers(self, higher_states):
        return torch.mean(higher_states, axis=0)

    def batch_init_story(self):
        s = torch.zeros((self.batch_size, self.input_units), requires_grad=True, device=self.device)
        return s
    
    def stop_grads(self, params_list):
        for params in params_list:
            params.requires_grad = False

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

    def predict_states(self, higher_state, policy=None):

        PLOT_TIMESTEPS = 240
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
            self.zero_grad()
        
        # Generate the video
        # clip = VideoClip(lambda t: env_frames[int(t)], duration=PLOT_TIMESTEPS)
        clip = ImageSequenceClip(env_frames, fps=24)
        clip.write_videofile("/mmfs1/gscratch/rao/vsathish/quals/plots/patch_predictions/"+"patch_predictions.mp4")
        # clip.write_gif("/mmfs1/gscratch/rao/vsathish/quals/plots/patch_predictions/"+"patch_predictions.gif", fps=20)
        

    def plot_predictions(self, canvas, cur_state, predictions, t):
        
        # Updated canvas
        alpha = 0.93
        x, y = cur_state[0]-1, cur_state[1]-1
        # canvas[cur_state] = 0
        for i in range(3):
            for j in range(3):
                if x+i >= 0 and x+i < 5 and y+j >= 0 and y+j < 5:
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
        



##################################
# Lower Level Transition Model 
##################################

class LowerRNN(nn.Module):
    
    def __init__(self, k, device, batch_size, inp, out):
        super(LowerRNN, self).__init__()
    
        self.top_down_weights = k
        self.device = device
        self.batch_size = batch_size
        self.input_shape = inp
        self.output_shape = out
        # self.hidden_size = 32
        self.hidden_size = 128
        self.decoder_size = 128
        self.num_layers = 1
        self.rnn = nn.RNN(input_size=self.top_down_weights+self.input_shape,\
                        hidden_size=self.hidden_size, num_layers=self.num_layers)
        # self.decoder1 = nn.Linear(self.hidden_size, self.decoder_size)
        # self.decoder2 = nn.Linear(self.decoder_size, self.output_shape)
        self.decoder = nn.Linear(self.hidden_size, self.output_shape)
        self.decoder1 = nn.Linear(128, 128)
        self.decoder2 = nn.Linear(128, 64)
        self.decoder3 = nn.Linear(64, self.output_shape)

    # Batch forward
    def forward(self, inputs, weights):

        inp = torch.cat((inputs, weights), dim=2)
        out, _ = self.rnn(inp)
        # output = F.relu(self.decoder(out))
        output = F.relu(self.decoder3(\
                        F.relu(self.decoder2(\
                            F.relu(self.decoder1(out))))))
        return output    

    def forward_inference(self, batch_input, weights, hidden):

        inp = torch.unsqueeze(torch.cat([batch_input, weights], dim=1), dim=0)
        # print(inp.shape, hidden.shape)
        out, h = self.rnn(inp, hidden)
        # output = torch.squeeze(F.relu(self.decoder(out)))
        output = F.relu(self.decoder3(\
                        F.relu(self.decoder2(\
                            F.relu(self.decoder1(out))))))
        output = torch.squeeze(output)
        return output, h

MODEL_PARAMS_OLD = {
    "input_units": 4,
    "output_units": 32,
    "data_in_dims": 13,
    "data_out_dims": 9,
    "input_timesteps": 25,
    "infer_lr": 0.1,
    "hyper_lr": 0.0005,
    "temporal_lr": 0.0005,
    "infer_max_iter": 20,
    "l2_lambda": 0.00001,

    "hypernet_layers": [128, 64, 64],

}
