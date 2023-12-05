import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

###########################################
# Network with learnable Stories
###########################################

class ActionEmbedding(nn.Module):
    def __init__(self, device, batch_size, timesteps=25):
        super(ActionEmbedding, self).__init__()
        self.input_units = 4
        self.output_units = 32
        self.data_in_dims = 13
        self.data_out_dims = 9
        self.input_timesteps = timesteps
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
        self.temporal = LowerRNN(self.output_units, device, \
                        batch_size, self.data_in_dims, self.data_out_dims)
        
        # Learning rates
        self.infer_lr = 0.1
        self.hyper_lr = 0.001
        self.temporal_lr = 0.001
        
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
        
        # This is the final weights for this batch of samples
        weights = self.hypernet(higher_state)
        # print(higher_state[0])
        temporal_batched_weights = weights.repeat(self.input_timesteps, 1, 1)
        # print("shapes : ", temporal_batch_input.shape, temporal_batch_output.shape, weights.shape)

        predicted_states = self.temporal(temporal_batch_input, temporal_batched_weights)
        errors = torch.pow(predicted_states-temporal_batch_output, 2)
        # print(errors.shape)
        errors = torch.sum(torch.sum(errors, axis=-1), axis=0)
        # print(errors.shape)
        errors = torch.mean(errors)
        
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
        
        hidden = torch.zeros((self.temporal.num_layers, self.batch_size, \
                self.temporal.hidden_size), device=self.device)
        
        for i in range(self.infer_max_iter):
            weights = self.hypernet(story)
            # print("Weights : ", weights.shape)
            hidden = hidden.detach()
            predicted_states, hidden = self.temporal.forward_inference(batch_input, weights, hidden)
            # print("Temporal next states : ", predicted_states.shape)
            
            loss = self.batch_errors(batch_output, predicted_states)+\
                self.l2_lambda * torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)
                # self.l1_lambda * torch.mean(torch.sum(torch.abs(story), axis=-1), axis=0)

            # if self.eval_mode:
            #     print("L2 val : ", torch.mean(torch.sum(torch.pow(story, 2), axis=-1), axis=0)\
            #         .detach().cpu().numpy(), "\t Loss : ", loss.detach().cpu().numpy())
            
            # print("loss : ", loss)
            # Backward - but only update story
            loss.backward()
            infer_optimizer.step()
            infer_optimizer.zero_grad()
            self.zero_grad()
            # print("backward pass done for step : ", i)
            
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

class LowerRNN(nn.Module):
    
    def __init__(self, k, device, batch_size, inp, out):
        super(LowerRNN, self).__init__()
    
        self.top_down_weights = k
        self.device = device
        self.batch_size = batch_size
        self.input_shape = inp
        self.output_shape = out
        self.hidden_size = 32
        self.num_layers = 1
        self.rnn = nn.RNN(input_size=self.top_down_weights+self.input_shape,\
                        hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_shape)

    # Batch forward
    def forward(self, inputs, weights):
        # print("in forward : " , inputs.shape, weights.shape)
        inp = torch.cat((inputs, weights), dim=2)
        # print(inp.shape)
        out, _ = self.rnn(inp)
        output = self.decoder(out)
        # print("final output : ", output.shape)    
        return output    

    def forward_inference(self, batch_input, weights, hidden):
        inp = torch.unsqueeze(torch.cat([batch_input, weights], dim=1), dim=0)
        # print(inp.shape, hidden.shape)
        
        out, h = self.rnn(inp, hidden)
        # print("post rnn : ", out.shape, h.shape)
        output = torch.squeeze(self.decoder(out))
        # print("final : ", output.shape)
        return output, h