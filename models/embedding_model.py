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
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


###########################################
# Network with learnable Stories
###########################################

MODEL_PARAMS = {
    "input_units": 32,
    "output_units": 32,
    "data_in_dims": 13,
    "data_out_dims": 9,
    "input_timesteps": 20,
    "infer_lr": 0.1,
    "hyper_lr": 0.001,
    "temporal_lr": 0.001,
    "infer_max_iter": 20,
    "l2_lambda": 0.0001,
    # "var_lambda": 1,
    # "center_lambda": 0.001,
    "hypernet_layers": [128, 256, 256],

}

INFER_PARAMS = {
    "input_units": 32,
    "output_units": 32,
    "data_in_dims": 13,
    "data_out_dims": 9,
    "input_timesteps": 25,
    "infer_lr": 0.1,
    "hyper_lr": 0.0005,
    "temporal_lr": 0.0005,
    "infer_max_iter": 20,
    "l2_lambda": 0.0001,
    # "var_lambda": 1,
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
        # self.var_lambda = params["var_lambda"]

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
                self.temporal.hidden_size), device=self.device)
        
        for i in range(self.infer_max_iter):
            weights = self.hypernet(story)
            # print("Weights : ", weights.shape)
            hidden = hidden.detach()
            predicted_states, hidden = self.temporal.forward_inference(batch_input, weights, hidden)
            # print("Temporal next states : ", predicted_states.shape)
            
            '''
            # Compute Loss.
            # Check if L2 norm gives better results
            # Check if variance/STD gives a better result
            '''
            loss = self.batch_errors(batch_output, predicted_states) + torch.sum(torch.var(story, axis=0)) +\
                    self.l2_lambda * torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)
                    
                # self.l1_lambda * torch.mean(torch.sum(torch.abs(story), axis=-1), axis=0)
            
            loss.backward()
            infer_optimizer.step()
            infer_optimizer.zero_grad()
            self.zero_grad()
            # print("backward pass done for step : ", i)
        
        if self.eval_mode and timestep % 20 == 0:
            print("Story shape and variance(STD) : ", story.shape, torch.sum(torch.std(story, axis=0)))
            print("L2 val : ", torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)\
                 .detach().cpu().numpy(), "\t Loss : ", loss.detach().cpu().numpy())
            
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

    def prediction_video(self, higher_state, policy=None):
        POMDP = (3, 3)
        START = (2, 2) 
        PLOT_TIMESTEPS = 50
        PLOT_BATCH_SIZE = 100
        canvas = -np.ones((5, 5)) # Initialize the cells with -1
        print("center : ", higher_state.shape)
        higher_state_batched = higher_state.repeat((PLOT_BATCH_SIZE, 1))
        print("batched center : ", higher_state_batched.shape)
        weights = self.hypernet(higher_state_batched)
        print("weights : ", weights.shape)
        # temporal_weights = weights.repeat((PLOT_TIMESTEPS, 1, 1))
        # print("temporal weights : ", temporal_weights.shape)

        # Use a batch of random initial states and same actions and average out the predicted states
        # Generate for each timestep
        # for timestep in range(PLOT_TIMESTEPS):
        #     state_input = torch.
        temporal_input = torch.rand((PLOT_TIMESTEPS, PLOT_BATCH_SIZE, self.data_in_dims), device=self.device)
        predicted_states = self.temporal(temporal_input, temporal_weights)
        

    def plot_predictions(self, canvas, cur_state, actions, predictions):
        

        cmap = colors.ListedColormap(['white', 'black', 'gray'])
        bounds = [-1,0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        
        ax.grid(which='major', axis='both', linestyle='-', color='darkgray', linewidth=2)
        # ax.set_xticks(np.arange(-0.5, 100.5, 1))
        # ax.set_yticks(np.arange(-0.5, 100.5, 1))

        plt.title("Drawing the Environment with higher states")
        ax.imshow(predictions, cmap=cmap, norm=norm)
        
        return mplfig_to_npimage(fig), cur_state
        



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
        self.hidden_size = 32
        self.decoder_size = 128
        self.num_layers = 1
        self.rnn = nn.RNN(input_size=self.top_down_weights+self.input_shape,\
                        hidden_size=self.hidden_size, num_layers=self.num_layers)
        # self.decoder1 = nn.Linear(self.hidden_size, self.decoder_size)
        # self.decoder2 = nn.Linear(self.decoder_size, self.output_shape)
        self.decoder = nn.Linear(self.hidden_size, self.output_shape)

    # Batch forward
    def forward(self, inputs, weights):
        # print("in forward : " , inputs.shape, weights.shape)
        inp = torch.cat((inputs, weights), dim=2)
        # print(inp.shape)
        out, _ = self.rnn(inp)
        # output = F.relu(self.decoder2(F.relu(self.decoder1(out))))
        output = F.relu(self.decoder(out))
        # output = self.decoder(out)
        # print("final output : ", output.shape)    
        return output    

    def forward_inference(self, batch_input, weights, hidden):
        inp = torch.unsqueeze(torch.cat([batch_input, weights], dim=1), dim=0)
        # print(inp.shape, hidden.shape)
        
        out, h = self.rnn(inp, hidden)
        # print("post rnn : ", out.shape, h.shape)
        # output = torch.squeeze(F.relu(self.decoder2(F.relu(self.decoder1(out)))))
        output = torch.squeeze(F.relu(self.decoder(out)))
        # output = torch.squeeze(self.decoder(out))
        # print("final : ", output.shape)
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
