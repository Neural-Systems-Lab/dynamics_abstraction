from math import e
import os
from re import S
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import itertools

'''
Plan in a compositional environment
'''
class AbstractPlanner():
    def __init__(self, config, base_configs, higher_state_model, \
                higher_state_util, lower_state_model, lower_action_model, \
                k, device):
        
        # Define global constants
        self.max_planning_steps = 20
        self.config = config    # Composition config
        self.base_configs = base_configs
        self.device = device
        self.k = k              # Planning horizon

        # Define APC model components
        self.higher_state_model = higher_state_model
        self.higher_state_util = higher_state_util
        self.lower_state_model = lower_state_model
        self.lower_action_model = lower_action_model

        self.higher_state_model.eval()
        self.lower_action_model.eval()

    def abstract_plan(self, env):
        
        # To actually plan, we find the fitness of each of the higher actions 
        # in the current higher state. We use manhattan distance as the fitness
        counter = 0
        action_seq = []
        print("############## START PLANNING ##############")
        for i in range(self.max_planning_steps):
            print(f"############## planning step {i+1} ##############")
        
            current_higher_state, higher_index = self.higher_state_util.get_abstract_state(env)
            if current_higher_state == None:
                continue
            print("In abstract planner : ", current_higher_state.shape, higher_index)

            # Find the best action in the current higher state
            best_action, teleport_ = self.best_abstract_action(higher_index, current_higher_state, env)
            env, actions, rewards, end = self.lower_action_model.execute_policy(env, torch.flatten(best_action))
            # sys.exit(0)
            counter += 1
            if end == 0:
                break
            else:
                env.step(teleport_) 
        
        print("############## PLANNING COMPLETE ##############")
        return env.episode_data, counter

    def best_abstract_action(self, higher_index, current_higher_state, env):
        # Find best abstract action with lookahead k
        abs_actions = self.base_configs[higher_index]["abs_actions"]
        teleports = self.base_configs[higher_index]["teleports"]
        
        # Check if the final goal is in the current state
        higher_token = env.get_higher_token()
        token_id = np.argmax(higher_token)
        metric_ = env.planning_metric(token_id)
        print("Planning Metric : ", metric_, higher_token)
        if metric_ == 0: # The goal is in the current higher state
            flag = 1
        else:
            flag = 0
        # abstract_action_list = []
        # teleports_list = []
        metrics = []
        
        for i in range(len(abs_actions)):
            action_ = torch.from_numpy(np.array(abs_actions[i])).float().flatten().to(self.device)

            if flag: # The goal is in the current higher state. So no need to plan
                print("######## Checking current higher action for goal ########")
                end = self.get_to_goal(env, action_)
                if end:
                    return action_, teleports[i][0]
                else:
                    continue

            # sys.exit(0)
            inputs = torch.cat((current_higher_state, action_), dim=0).unsqueeze(0)
            next_state = self.higher_state_model(inputs)
            print(current_higher_state, next_state, i)
            next_state_token = torch.squeeze(next_state, dim=0).detach().cpu().numpy()[4:]
            token_id = np.argmax(next_state_token)
            metric_ = env.planning_metric(token_id)
            # if metric_ == 0:
                # The goal is in the current higher state
            metrics.append(metric_)
            print("model ran successfully : ", next_state_token, metrics)

        # Best action is the one with the lowest metric
        best_action_idx = np.argmin(metrics)
        teleport_ = teleports[best_action_idx][0]
        best_action = torch.from_numpy(np.array(abs_actions[best_action_idx])).float().flatten().to(self.device)
        return best_action, teleport_

    def get_to_goal(self, env, higher_action):
        # We use the lower state model to get to the goal
        current_state = env.state
        print("Starting policy check : ", current_state)
        env, actions, rewards, end = self.lower_action_model.execute_policy(env, higher_action, record_step=False)
        print("Ending policy check. Final state : ", env.state, " Goal : ", env.goal)
        if env.state == env.goal:
            env.state = current_state
            end = 1
        else:
            env.state = current_state
            end = 0
        
        return end

'''
Plan without higher world model
'''
class FlatPlanner():
    def __init__(self, lookahead):
        self.k = lookahead % 5  # Make sure lookahead is not too large
        # self.device = device
        self.max_planning_steps = 25
    
    def plan(self, env):
        
        counter = 0
        action_seq = []
        for i in range(self.max_planning_steps):
            # Assume a perfect transition model
            opt_action = self.optimal_transition_(env)
            opt_action = opt_action[0]
            # print("Optimal Action : ", opt_action)
            _, _, end = env.step(opt_action)
            action_seq.append(opt_action)
            counter += 1
            if end == 0:
                break
        
        return action_seq, counter


    def optimal_transition_(self, env):
        # current_state = env.state
        metrics = []
        action_seq = list(itertools.product([0, 1, 2, 3], repeat=self.k))

        for action_tuples in action_seq:
            current_state = env.state
            for action in action_tuples:
                env.step(action, record_step=False)
            metrics.append(self.manhattan_distance_(env))
            env.state = current_state

        print(metrics)
        optimal_idx = np.argmin(np.array(metrics))
        print("optimal_action :", optimal_idx)
        return action_seq[optimal_idx]


    def manhattan_distance_(self, env):
        state = env.state
        goal = env.goal
        print("State and Goal : ", state, goal)
        return np.abs(state[0]-goal[0]) + np.abs(state[1]-goal[1])

