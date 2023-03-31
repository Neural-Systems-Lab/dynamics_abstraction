# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch.nn as nn
import torch
import torch.nn.functional as F



class LinearModule(nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.hidden1 = nn.Linear(15, 11)

    def forward(self, x):
        
        next_state = F.relu(self.hidden1(x))
        return next_state

        
class NonLinearModule(nn.Module):
    def __init__(self):
        super(NonLinearModule, self).__init__()
        self.hidden1 = nn.Linear(15, 64)
        self.hidden2 = nn.Linear(64, 128)
        self.hidden3 = nn.Linear(128, 11)
    
    def forward(self, x):
        next_state = F.relu(self.hidden3(F.relu(self.hidden2(F.relu(self.hidden1(x))))))
        return next_state
    
        
class RecurrentModule(nn.Module):
    
    def __init__(self):
        super(RecurrentModule, self).__init__()
        self.rnn = nn.RNN(input_size=15, hidden_size=64, \
                        num_layers=1, nonlinearity='relu',\
                        batch_first=True)
        self.decoder = nn.Linear(64, 11)

    def forward(self, x):
        # print("Input : ", x.shape, h.shape)
        out, h = self.rnn(x)
        # print("Out before : ", out.shape, h.shape)
        out = F.relu(self.decoder(out))
        # print("Out after : ", out.shape, h.shape)
        return out, h
        
        
###########################################
# The 2 layer network with fixed s1 and s2
###########################################
class HypernetMatrix(nn.Module):
    
    def __init__(self, device):
        super(HypernetMatrix, self).__init__()
        self.k = 8
        self.device = device
        self.hypernet = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.k)
        )
        
        self.temporal = ModulatedMatrix(self.k, device)
        
    def forward(self, higher_state, trajectory):
        
        # Stays the same for current input sequence
        higher_state = torch.tensor(higher_state).type(torch.FloatTensor).to(self.device)
        weights = self.hypernet(higher_state)
        
        # print("WEIGHTSSSSS : ", weights.detach().cpu().numpy())
        
        errors = torch.zeros(1, requires_grad=True).to(self.device)
        for state, action, next_state in trajectory:
            
            predicted_states = self.temporal(state, action, weights)
            # print(predicted_states.detach().cpu().numpy(), next_state)
            
            next_state = torch.tensor(next_state).type(torch.FloatTensor).to(self.device)    
            errors += self.prediction_errors(predicted_states, next_state)
        
        self.weights_ = weights
        return errors/len(trajectory)

    def prediction_errors(self, predicted, ground_truth):
        err = (torch.squeeze(predicted)-ground_truth)**2
        return torch.mean(err)

class ModulatedMatrix(nn.Module):
    
    def __init__(self, k, device):
        super(ModulatedMatrix, self).__init__()
    
        self.k = k
        self.device = device
        self.temporal_ = nn.Parameter(torch.randn(self.k, \
                    15, 11, requires_grad=True))

    def forward(self, state, action, modulation_weights):
        input_ = torch.cat((torch.tensor(state), torch.tensor(action)), 0).type(torch.FloatTensor).to(self.device)
        
        # Weighted sum of temporal params
        v_matrix = torch.matmul(modulation_weights, self.temporal_.reshape(self.k, -1))
        v_matrix = v_matrix.reshape(1, 15, 11)
        
        # Next state
        next_state = F.relu(torch.matmul(input_, v_matrix))
        
        return next_state



###########################################
# Network with learnable S1 and S2
###########################################

class LearnableStory(nn.Module):
    def __init__(self, device):
        super(LearnableStory, self).__init__()
        self.input_units = 2
        self.output_units = 4
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
        self.infer_lr = 0.01
        self.hyper_lr = 0.001
        self.temporal_lr = 0.001
    
        
        # Other vars
        self.infer_max_iter = 25
    
    def forward(self, trajectory, eval="False"):
        
        # Stays the same for current input sequence
        self.higher_state = self.inference(trajectory)
        weights = self.hypernet(self.higher_state)
        print("Higher State : ", self.higher_state.detach().cpu().numpy())
        
        print("WEIGHTSSSSS : ", weights.detach().cpu().numpy())
        
        errors = torch.zeros(1, requires_grad=True).to(self.device)
        
        # losses = 0
        predicted_states_list = []
        for state, action, next_state in trajectory:

            predicted_states = self.temporal(state, action, weights)
            next_state = torch.tensor(next_state).type(torch.FloatTensor).to(self.device)    
            errors += self.prediction_errors(predicted_states, next_state)
            
            predicted_states_list.append(predicted_states.detach().cpu().numpy())
            # losses += loss.detach().cpu().numpy()
            

        
        self.weights_ = weights
        return errors/len(trajectory), predicted_states_list, self.higher_state.cpu().numpy()

    def prediction_errors(self, predicted, ground_truth):
        err = (torch.squeeze(predicted)-ground_truth)**2
        return torch.mean(err)
    
    def inference(self, trajectory):
        # Fix temporal and hypernet weights.
        # Train only the latent
        
        story = self.init_story()
        infer_optimizer = torch.optim.Adam([story], self.infer_lr)
        # converged = False
        i = 0
        
        for i in range(self.infer_max_iter):

            # for state, action, next_state in trajectory:
            state, action, next_state = trajectory[0]
            
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