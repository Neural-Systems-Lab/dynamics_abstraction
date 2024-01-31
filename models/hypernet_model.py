import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

###########################################
# Network with learnable Stories
###########################################

class LearnableHypernet(nn.Module):
    def __init__(self, device, batch_size):
        super(LearnableHypernet, self).__init__()
        self.input_units = 4
        self.output_units = 32
        self.device = device
        self.batch_size = batch_size
        
        # Model params
        self.hypernet = nn.Sequential(
            nn.Linear(self.input_units, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_units)
        )
        self.temporal = ModulatedMatrix(self.output_units, device, batch_size)
        
        # Learning rates
        self.infer_lr = 0.05
        self.hyper_lr = 0.001
        self.temporal_lr = 0.005
        
        # Other vars
        self.infer_max_iter = 10
        self.l2_lambda = 0.0001
        self.l1_lambda = 0.0

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
        
        higher_state_list.append(self.batch_init_story().clone().detach().cpu().numpy())


        for t in range(len(temporal_batch_input)):
        # for state, action, next_state in trajectory:
            
            higher_state = self.batch_inference(temporal_batch_input[t], \
                            temporal_batch_output[t], higher_state)
            
            if eval_mode:
                higher_state_list.append(torch.squeeze(higher_state).detach().cpu().numpy())
        
        weights = self.hypernet(higher_state)
        
        for t in range(len(temporal_batch_input)):
            
            predicted_states = self.temporal(temporal_batch_input[t], weights)
            errors += self.batch_errors(temporal_batch_output[t], predicted_states)
            # print(errors)
            
            if eval_mode:
                # print("saving higher states")
                predicted_states_list.append(predicted_states.detach().cpu().numpy())
                # higher_state_list.append(torch.squeeze(higher_state).detach().cpu().numpy())
                # weights_list.append(weights.detach().cpu().numpy())
        
        # errors = errors/len(temporal_batch_input)
        
        # print("######### Predicting the states #########")
        # print("Shapes : ", predicted_states.shape, temporal_batch_output.shape)
        # print("Predicted states : ", predicted_states[0])
        # print("True states : ", temporal_batch_output[-1][0])

        if eval_mode:
            return errors.detach().cpu().numpy(), \
                predicted_states_list, higher_state_list#, weights_list
                
        # return errors/len(temporal_batch_input)
        return errors
    
    def batch_inference(self, batch_input, batch_output, previous_story=None):
        
        if previous_story == None:
            story = self.batch_init_story()
        else:
            story = previous_story.clone()
            story.requires_grad = True
        
        # Is this optimizer definition correct for batched story?
        infer_optimizer = torch.optim.SGD([story], self.infer_lr)
        # print(story, infer_optimizer)
        
        for i in range(self.infer_max_iter):
            weights = self.hypernet(story)
            # print("Weights : ", weights.shape)
            
            predicted_states = self.temporal(batch_input, weights)
            # print("Temporal next states : ", predicted_states.shape)
            
            loss = self.batch_errors(batch_output, predicted_states)+\
                self.l2_lambda * torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)
                # self.l1_lambda * torch.mean(torch.sum(torch.abs(story), axis=-1), axis=0)
            
            # Backward - but only update story
            loss.backward()
            infer_optimizer.step()
            infer_optimizer.zero_grad()
            self.zero_grad()
            # print("backward pass")
    
        if self.eval_mode:
            print("L2 val : ", torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)\
                .detach().cpu().numpy(), "\t Loss : ", loss.detach().cpu().numpy())

        return story.clone().detach()

    def batch_errors(self, true, predicted):  
        err = torch.pow(true-predicted, 2)
        err = torch.sum(err, axis=-1)
        err = torch.mean(err, axis=0)
        return err
    
    def batch_init_story(self):
        s = torch.zeros((self.batch_size, self.input_units), requires_grad=True, device=self.device)
        return s


    ##############
    # Samplewise
    ##############

'''
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
    
'''    

##################################
# Lower Level Transition Model 
##################################

class ModulatedMatrix(nn.Module):
    
    def __init__(self, k, device, batch_size):
        super(ModulatedMatrix, self).__init__()
    
        self.k = k
        self.device = device
        self.batch_size = batch_size
        self.input_shape = 13
        self.output_shape = 9
        self.temporal_ = nn.Parameter(torch.randn(self.k, \
                    self.input_shape, self.output_shape, requires_grad=True))

    '''
    def single_forward(self, state, action, modulation_weights):
        input_ = torch.cat((torch.tensor(state), torch.tensor(action)), 0).type(torch.FloatTensor).to(self.device)
        
        # Weighted sum of temporal params
        v_matrix = torch.matmul(modulation_weights, self.temporal_.reshape(self.k, -1))
        v_matrix = v_matrix.reshape(1, 15, 11)
        
        # Next state
        next_state = F.relu(torch.matmul(input_, v_matrix))
        
        return next_state
    '''
    # Batch forward
    def forward(self, inputs, weights):
        
        # print(self.temporal_)
        # sys.exit(0)
        v_matrix = torch.matmul(weights, self.temporal_.reshape(self.k, -1))
        # print("lower transition : ", v_matrix.shape)
        v_matrix = v_matrix.reshape(self.batch_size, self.input_shape, self.output_shape)
        inputs = torch.unsqueeze(inputs, 1)
        # print("Shapes : ", v_matrix.shape, inputs.shape)
        next_states = F.relu(torch.bmm(inputs, v_matrix))
        
        return torch.squeeze(next_states)
