'''
For a given mini-environment and a set of sub-goals, extract policy data for each subgoal
'''

import numpy as np
import random
import sys

import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.distributions import Categorical
from dataloaders.parallel_envs import ParallelEnvironments


class LowerPolicyTrainer(nn.Module):
    def __init__(self, device, batch_size, max_timesteps, env_handler=None):
        '''
        define model constants and hyperparams
        '''
        super().__init__()

        # Batched environment class
        self.env = env_handler

        # Logger for plots
        self.epoch_rewards = []
        self.epoch_critic_losses = []
        self.epoch_actor_losses = []

        # Model params
        self.top_down_ip_sz = 16
        self.top_down_op_sz = 64
        self.obs_space = 9
        self.action_space = 4
        self.max_timesteps = max_timesteps
        self.device = device
        self.batch_size = batch_size
        self.l2_lambda = 0

        # Top down action hypernet H_a
        self.hypernet = ActionHypernetwork(self.top_down_ip_sz, self.top_down_op_sz)

        # Lower level policy f_a
        self.policy = PolicyRNN(self.top_down_op_sz, device, batch_size,
                            self.obs_space, self.action_space).to(device)
        
        self.critic = StateValueCritic(self.obs_space, self.top_down_op_sz)

    def forward(self):
        '''
        Take actions based on current policy and send these actions to data handler
        higher_actions.shape = (batch_sz, num_features * num_vectors)
        '''
        vars = {
            "log_probs":[], "mask":[], "reward":[], "l2_logits":[], "critic":[]
        }

        # Hypernet
        higher_actions = self.env.get_higher_actions()
        top_down_weights = self.hypernet(higher_actions)
        observations = self.env.batch_reset()
        self.policy.init_hidden()

        for i in range(self.max_timesteps):
            
            # On-Policy Actions
            action_logits = self.policy(observations, top_down_weights)
            distributions = Categorical(logits=action_logits)
            actions = distributions.sample()
            log_probs = distributions.log_prob(actions)
            
            if i % 10 == 0:
                print(action_logits[2].detach().cpu().numpy())
                # print(action_logits[20].detach().cpu().numpy())
            # L2 constraint on action logits
            l2 = torch.sum(torch.square(action_logits), axis=1)

            # Acting in the Environment
            next_obs, rewards, mask = self.env.batch_step(actions)

            # Critic values
            values = self.critic(next_obs, top_down_weights)

            # Update state
            observations = next_obs

            # Track the variables for updates
            vars["log_probs"].append(log_probs)
            vars["l2_logits"].append(l2)
            vars["reward"].append(rewards)
            vars["mask"].append(mask)
            vars["critic"].append(values)
    
        return self.compute_losses(vars)

    def compute_losses(self, vars):
        rewards = torch.stack(vars["reward"])
        l2 = torch.stack(vars["l2_logits"])
        mask = torch.stack(vars["mask"])
        # print(mask)
        log_probs = torch.squeeze(torch.stack(vars["log_probs"]))
        critic_values = torch.squeeze(torch.stack(vars["critic"]))
        # print("rewards : ", rewards.shape)
        # print("rewards : ", torch.sum(rewards, axis=0))
        self.epoch_rewards.append(torch.mean(torch.sum(rewards, axis=0)).detach().cpu().numpy())
        
        discounted_returns = self.discounted_rewards(rewards) #- critic_values.detach()
        log_probs = log_probs * mask

        # print("returns : ", torch.t(discounted_returns), discounted_returns.shape)
        # print("Log probs : ", log_probs.detach().cpu().numpy().T)
        # intermediate = torch.sum(log_probs * discounted_returns, axis=0)
        # print("here1:", intermediate)
        # print(torch.t(mask), mask.shape)

        actor_loss = - torch.mean(torch.sum(log_probs * discounted_returns, axis=0)) #+ torch.mean(l2 * self.l2_lambda) 
        critic_loss = torch.mean(torch.sum(torch.square(discounted_returns - critic_values), axis=0))
        
        self.epoch_actor_losses.append(actor_loss.detach().cpu().numpy())
        self.epoch_critic_losses.append(critic_loss.detach().cpu().numpy())
        
        # print("Intermediate returns : \n", l2.shape, mask.shape, log_probs.shape)
        # discounted_returns = discounted_returns - critic_values.detach()
        return actor_loss, critic_loss

    def discounted_rewards(self, rewards):
        GAMMA = 0.99
        R = 0
        eps = np.finfo(np.float32).eps.item()
        returns = []
        
        rewards_ = rewards.detach().cpu().numpy()
        # print("raw rewards = ", rewards_)
        for r in rewards_[::-1]:
            R = r + GAMMA*R
            returns.insert(0, R)
        
        returns = torch.from_numpy(np.array(returns)).to(self.device)
        discounted_returns = (returns - returns.mean(axis=0)) / (returns.std(axis=0) + eps)

        return discounted_returns
    
    # @torch.no_grad
    def execute_policy(self, env, higher_actions):
        '''
        Inference of optimal actions from learnt policy
        '''
        print("################## BEGIN EXECUTE POLICY ##################")
        actions = []
        rewards = []
        states_list = []
        state = torch.tensor(env.get_pomdp_state()).to(self.device, dtype=torch.float32)
        state = torch.unsqueeze(state, dim=0)
        states_list.append(env.get_state())
        # print("State action shape : ", state.shape, higher_actions.shape)
        top_down_weights = self.hypernet(higher_actions)
        top_down_weights = torch.unsqueeze(top_down_weights, dim=0)
        print("Top down weights : ", top_down_weights.shape)
        self.policy.init_hidden()
        for i in range(self.max_timesteps):
            
            # Here both state and weights are expected to have a batch dimension
            action_logits = self.policy(state, top_down_weights)
            action = torch.argmax(action_logits)
            next_state, reward, end = env.step(action.detach().cpu().numpy())
            state = torch.tensor(next_state).to(self.device, dtype=torch.float32)
            state = torch.unsqueeze(state, dim=0)
            states_list.append(env.get_state())
            
            actions.append(action.detach().cpu().numpy())
            
            rewards.append(reward)
            
            if end == 0:
                break
            
            # Add a constraint that if the agent is landing on the same state
            # or hitting a wall repeatedly for 3 times, then end the policy.
            #  Note that same action can occur multiple times.
            # print(actions)
            if i > 4 and (states_list[-1] == states_list[-2]) and \
                (states_list[-1] == states_list[-3]) and \
                (states_list[-1] == states_list[-4]) and \
                actions[-1] == actions[-2] == actions[-3] :
                print("Exiting sub-policy. Repeated state")
                break
            
            # If I reach a junction, this sub-policy ends
            elif len(env.get_higher_token()) > 1:
                print("Exiting sub-policy. Option completed")
                break

        return env, actions, rewards, end


class ActionHypernetwork(nn.Module):
    def __init__(self, top_down_ip_sz, top_down_op_sz):
        super().__init__()
        
        self.action_hypernet = nn.Sequential(
            nn.Linear(top_down_ip_sz, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, top_down_op_sz)
        )
    
    def forward(self, top_down_inputs):
        
        top_down_outputs = self.action_hypernet(top_down_inputs)
        # print(top_down_outputs.detach().cpu().numpy())

        return top_down_outputs


class StateValueCritic(nn.Module):
    def __init__(self, obs_space, top_down_sz):
        super().__init__()

        self.obs_space = obs_space
        self.top_down_sz = top_down_sz

        self.critic = nn.Sequential(
            nn.Linear(obs_space+top_down_sz, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, observations, top_down_weights):
        
        inputs = torch.cat((observations, top_down_weights), dim=1)
        value = self.critic(inputs)

        return value


class PolicyRNN(nn.Module):
    '''
    Lower level policy conditioned on top down weights
    '''
    def __init__(self, top_down_units, device, batch_size, inp, out):
        super().__init__()

        self.top_down_units = top_down_units
        self.device = device
        self.batch_size = batch_size
        self.input_shape = inp
        self.output_shape = self.num_actions = out
        
        self.rnn_hidden = 64
        self.hidden2 = 128
        self.hidden3 = 64
        self.num_layers = 1

        self.rnn = nn.RNN(input_size=self.top_down_units+self.input_shape,
                        hidden_size=self.rnn_hidden, 
                        num_layers=self.num_layers,
                        nonlinearity='tanh',
                        dropout=0.0)

        self.decoder1 = nn.Linear(self.rnn_hidden, self.hidden2)
        self.decoder2 = nn.Linear(self.hidden2, self.output_shape)
        self.hidden_units = torch.zeros(1, self.batch_size, self.rnn_hidden).to(self.device)

    def init_hidden(self):
        self.hidden_units = torch.zeros(1, self.batch_size, self.rnn_hidden).to(self.device)

    def forward(self, observations, top_down_weights):
        '''
        batch forward function
        '''
        # print(observations.shape, top_down_weights.shape)
        inp = torch.unsqueeze(torch.cat((observations, top_down_weights), dim=1), dim=0)
        # print("Input and hidden shape after concat and unsqueeze: ", \
        #     inp.shape, self.hidden_units.shape)
        out, new_hidden = self.rnn(inp, self.hidden_units)
        out = F.relu(self.decoder1(out))
        action_logits = self.decoder2(out)
        # print(f"Action probs: ", action_logits.shape)
        self.hidden_units = new_hidden

        return torch.squeeze(action_logits)
    