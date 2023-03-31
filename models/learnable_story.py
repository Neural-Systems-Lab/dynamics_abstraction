import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

class ModulatedMatrix(nn.Module):
    
    def __init__(self, k, device):
        super(ModulatedMatrix, self).__init__()
    
        self.k = k
        self.device = device
        self.input_shape = 15
        self.output_shape = 11
        self.temporal_ = nn.Parameter(torch.randn(self.k, \
                    self.input_shape, self.output_shape, requires_grad=True))

    def single_forward(self, state, action, modulation_weights):
        input_ = torch.cat((torch.tensor(state), torch.tensor(action)), 0).type(torch.FloatTensor).to(self.device)
        
        # Weighted sum of temporal params
        v_matrix = torch.matmul(modulation_weights, self.temporal_.reshape(self.k, -1))
        v_matrix = v_matrix.reshape(1, 15, 11)
        
        # Next state
        next_state = F.relu(torch.matmul(input_, v_matrix))
        
        return next_state

    def forward(self, inputs, weights):
        
        v_matrix = torch.matmul(weights, self.temporal_.reshape(self.k, -1))
        v_matrix = v_matrix.reshape(self.input_shape, self.output_shape)
        next_states = F.relu(torch.matmul(inputs, v_matrix))
        
        return next_states


###########################################
# Network with learnable S1 and S2
###########################################

class LearnableStory(nn.Module):
    def __init__(self, device, batch_size):
        super(LearnableStory, self).__init__()
        self.input_units = 16
        self.output_units = 16
        self.device = device
        
        # Model params
        self.hypernet = nn.Sequential(
            nn.Linear(self.input_units, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_units)
        )
        self.temporal = ModulatedMatrix(self.output_units, device)
        
        # Learning rates
        self.infer_lr = 0.1
        self.hyper_lr = 0.005
        self.temporal_lr = 0.005
        
        # Other vars
        self.infer_max_iter = 10

    ####################
    # Batched functions
    ####################
    
    def forward(self, batch_trajectory_input, batch_trajectory_output):
    
        errors = torch.zeros(1, requires_grad=True).to(self.device)
        predicted_states_list = []
        higher_state = None
        
        for timestep in range(len(batch_trajectory_input)):
        # for state, action, next_state in trajectory:
            
            higher_state = self.batch_inference(batch_input, batch_output, higher_state)
            weights = self.hypernet(higher_state)
            
            predicted_states = self.temporal(state, action, weights)
            next_state = torch.tensor(next_state).type(torch.FloatTensor).to(self.device)    
            errors += self.prediction_errors(predicted_states, next_state)
            
            predicted_states_list.append(predicted_states.detach().cpu().numpy())
            # losses += loss.detach().cpu().numpy()
            
        self.weights_ = weights
        return errors/len(trajectory), predicted_states_list, higher_state.cpu().numpy()

    def batch_inference(self, batch_input, batch_output, previous_story=None):
        
        if previous_story == None:
            story = self.batch_init_story()
        else:
            story = previous_story.clone()
            story.requires_grad = True
        
        # Is this optimizer definition correct for batched story?
        infer_optimizer = torch.optim.Adam([story], self.infer_lr)
        print(story, infer_optimizer)
        sys.exit(0)
        
        for i in range(self.infer_max_iter):
            weights = self.hypernet(story)
            predicted_states = self.temporal(batch_input, weights)
            loss = self.batch_errors(batch_output, predicted_states)
        
            # Backward - but only update story
            loss.backward()
            infer_optimizer.step()
            infer_optimizer.zero_grad()
            self.zero_grad()
        
        return story.clone().detach()

    def batch_errors(self, true, predicted):  
        return torch.mean(torch.sum(torch.pow(true-predicted, 2), axis=-1), axis=0)
    
    def batch_init_story(self, batch_size):
        s = torch.zeros((batch_size, self.input_units), requires_grad=True, device=self.device)
        return s


    ##############
    # Samplewise
    ##############


    def samplewise_forward(self, batch_trajectory_input, batch_trajectory_output):
    
        errors = torch.zeros(1, requires_grad=True).to(self.device)
        predicted_states_list = []
        higher_state = None
        
        for timestep in range(len(batch_trajectory_input)):
        # for state, action, next_state in trajectory:
            
            higher_state = self.inference((state, action, next_state), higher_state)
            weights = self.hypernet(higher_state)
            
            predicted_states = self.temporal(state, action, weights)
            next_state = torch.tensor(next_state).type(torch.FloatTensor).to(self.device)    
            errors += self.prediction_errors(predicted_states, next_state)
            
            predicted_states_list.append(predicted_states.detach().cpu().numpy())
            # losses += loss.detach().cpu().numpy()
            
        
        self.weights_ = weights
        
        
        return errors/len(trajectory), predicted_states_list, higher_state.cpu().numpy()

    def prediction_errors(self, predicted, ground_truth):
        err = (torch.squeeze(predicted)-ground_truth)**2
        return torch.sum(err)
    
    def inference(self, triplet, previous_story=None):
        # Fix temporal and hypernet weights.
        # Train only the latent
        
        if previous_story == None:
            story = self.init_story()
        else:
            story = previous_story.clone()
            story.requires_grad = True
        
        infer_optimizer = torch.optim.Adam([story], self.infer_lr)
        # converged = False
        state, action, next_state = triplet
        
        for i in range(self.infer_max_iter):
            
            # Forward
            weights = self.hypernet(story)
            predicted_states = self.temporal(state, action, weights)
            next_state = torch.tensor(next_state).type(torch.FloatTensor).to(self.device)
            loss = self.prediction_errors(predicted_states, next_state)

            # Backward - but only update story
            loss.backward()
            infer_optimizer.step()
            infer_optimizer.zero_grad()
            self.zero_grad()
                
            # print("@@@@ Story @@@@ : ", story.detach().cpu().numpy())

        return story.clone().detach()

    def init_story(self):
        s = torch.zeros((self.input_units), requires_grad=True, device=self.device)
        return s
    
    
    
    