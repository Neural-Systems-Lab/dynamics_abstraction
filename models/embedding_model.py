import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

###########################################
# Network with learnable Stories
###########################################

class LearnableStory(nn.Module):
    def __init__(self, device, batch_size):
        super(LearnableStory, self).__init__()
        self.input_units = 4
        self.output_units = 32
        self.data_in_dims = 13
        self.data_out_dims = 9
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
        self.temporal = ModulatedMatrix(self.output_units, device, \
                        batch_size, self.data_in_dims, self.data_out_dims)
        
        # Learning rates
        self.infer_lr = 0.1
        self.hyper_lr = 0.001
        self.temporal_lr = 0.005
        
        # Other vars
        self.infer_max_iter = 10
        self.l2_lambda = 0.001
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
                print("saving higher states")
                predicted_states_list.append(predicted_states.detach().cpu().numpy())
                # higher_state_list.append(torch.squeeze(higher_state).detach().cpu().numpy())
                # weights_list.append(weights.detach().cpu().numpy())
        
        errors = errors/len(temporal_batch_input)
        
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

            if self.eval_mode:
                print("L2 val : ", torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)\
                    .detach().cpu().numpy(), "\t Loss : ", loss.detach().cpu().numpy())
            
            # Backward - but only update story
            loss.backward()
            infer_optimizer.step()
            infer_optimizer.zero_grad()
            self.zero_grad()
            # print("backward pass")
            
        return story.clone().detach()

    def batch_errors(self, true, predicted):  
        err = torch.pow(true-predicted, 2)
        err = torch.sum(err, axis=-1)
        err = torch.mean(err, axis=0)
        return err
    
    def batch_init_story(self):
        s = torch.zeros((self.batch_size, self.input_units), requires_grad=True, device=self.device)
        return s


##################################
# Lower Level Transition Model 
##################################

class ModulatedMatrix(nn.Module):
    
    def __init__(self, k, device, batch_size, inp, out):
        super(ModulatedMatrix, self).__init__()
    
        self.top_down_weights = k
        self.device = device
        self.batch_size = batch_size
        self.input_shape = inp
        self.output_shape = out
        self.hidden_size = 32
        self.temporal_ = nn.RNN(input_size=self.top_down_weights+self.input_shape,\
                        hidden_size=self.hidden_size, num_layers=2)

    # Batch forward
    def forward(self, inputs, weights):
        
        inp = torch.cat((inputs, weights), dims=2)
        out, _ = self.temporal_
        
        
        # # print(self.temporal_)
        # # sys.exit(0)
        # v_matrix = torch.matmul(weights, self.temporal_.reshape(self.k, -1))
        # # print("lower transition : ", v_matrix.shape)
        # v_matrix = v_matrix.reshape(self.batch_size, self.input_shape, self.output_shape)
        # inputs = torch.unsqueeze(inputs, 1)
        # # print("Shapes : ", v_matrix.shape, inputs.shape)
        # next_states = F.relu(torch.bmm(inputs, v_matrix))
        
        # return torch.squeeze(next_states)