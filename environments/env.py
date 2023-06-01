'''
Created by: Vishwas Sathish
Date: Jan 31, 2023
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
# import cv2

from .configs import *


class SimpleGridEnvironment:
    def __init__(self, config, goal, start_states):
        
        # Create the canvas with walls and a single goal state
        self.rows, self.cols, self.walls, self.env_name = \
            config["rows"], config["cols"], config["walls"], config["name"]

        self.config = config
        self.board = np.zeros([self.rows, self.cols])
        self.actions = {0:"Right", 1:"Left", 2:"Down", 3:"Up"}
        self.goal = goal
        
        x, y = goal
        self.board[x, y] = GOAL_PIXEL
        
        self.valid_start_positions = []
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in self.walls:
                    # (i, j) != self.goal:
                    self.valid_start_positions.append((i, j))
        
        # print("##### START STATESSS #####")
        # print(self.valid_start_positions)
        
        # if goal in self.valid_start_positions:
        #     # print(f"removing {goal} from start positions")
        #     self.valid_start_positions.remove(goal)
        
        self.one_hot_mapping = {}
        counter = 0
        for i in range(self.rows):
            for j in range(self.cols):

                one_hot = np.zeros((self.rows*self.cols))
                one_hot[counter] = 1
                self.one_hot_mapping[(i, j)] = one_hot
                
                counter +=1

        # Add walls and goal state to canvas
        for x ,y in self.walls:
            self.board[x, y] = WALL_PIXEL
        

        
        # Will contain agent position for each step of each episode
        # Each step is (s, a, s', r)
        self.historic_data = []
        self.episode_data = []


    def reset(self, start_position=None):
        # reset to some valid start state
        # Store and reset all the episodic variables
        if len(self.episode_data) > 0:
            self.historic_data.append(self.episode_data)
            self.episode_data = []
        
        
        # Give start state else random start
        if start_position != None:
            try:
                assert start_position != self.goal
                self.state = start_position
                
            except:
                print(f"Start State {start_position} is the same as goal state! Starting randomly")
                idx = np.random.randint(0, len(self.valid_start_positions))
                self.state = self.valid_start_positions[idx]
        
        else:
            idx = np.random.randint(0, len(self.valid_start_positions))
            self.state = self.valid_start_positions[idx]
        
        # I need to figure out how to jump higher level states
        # print(self.one_hot_mapping, self.state)
        # print(self.one_hot_mapping[self.state])
        return self.one_hot_mapping[self.state].flatten()
    
    def step(self, action):
        
        next_state = self.next_position(action)
        reward, end = self.reward_function(next_state)
        
        self.episode_data.append((self.state, action, next_state, reward))
        self.state = next_state
        
        return self.one_hot_mapping[self.state].flatten(), reward, end
        
    def next_position(self, action):
        # print(action)
        if action == 0:
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1)
        elif action == 3:
            nxtState = (self.state[0], self.state[1] + 1)
        
        # if next state legal
        if (nxtState[0] >= 0) and (nxtState[0] <= (self.rows -1)):
            if (nxtState[1] >= 0) and (nxtState[1] <= (self.cols -1)):
                if nxtState not in self.walls:
                    return nxtState
        
        return self.state
    
    def reward_function(self, state):
        # assert self.goal in self.valid_goals
        
        # Returns reward and termination condition
        # Any transition not to goal gets -1 step reward
        # I use termination key as mask hence interchanged
        if state == self.goal:
            return GOAL_REWARD, 0
        else:
            return STEP_REWARD, 1
        
    def print_board(self):
        # self.board[self.state] = POS_PIXEL
        for i in range(0, self.rows):
            print('---------------')
            out = '| '
            for j in range(0, self.cols):
                
                if (i, j) == self.goal:
                    token = 'G'
                elif (i, j) == self.state:
                    token = '*'
                elif self.board[i, j] == WALL_PIXEL:
                    token = 'X'
                elif self.board[i, j] == EMPTY_PIXEL:
                    token = '0'
                out += token + ' | '
            print(out)
        print('---------------')
    
    def print_policy_map(self):
        '''
        Ideally called after an episode has ended
        '''
        _board = self.board.tolist()
        
        # Fill map with actions
        for data in self.episode_data:
            x, y = data[0]
            print("state : ", x, y)
            act = data[1]
            _board[x][y] = self.actions[act]
        
        # Print map
        for i in range(self.rows):
            print('---------------')
            out = '| '
            for j in range(self.cols):
                
                if (i, j) == self.goal:
                    token = 'G'
                elif _board[i][j] == WALL_PIXEL:
                    token = 'X'
                elif _board[i][j] == EMPTY_PIXEL:
                    token = '0'
                else:
                    # Try priting the action taken
                    token = _board[i][j]
                    
                out += token + ' | '
            print(out)
        print('---------------')
            
        
    def plot_board(self):
        plt.clf()
        plt.close()
        _board = self.board.copy()
        # _board[self.state] = POS_PIXEL
        cmap = colors.ListedColormap(['black', 'gray', 'green'])
        bounds=[0,1,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax = plt.figure(figsize=(6, 6))
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        plt.title("Environment - "+str(self.config["id"]))
        plt.imshow(_board, cmap=cmap, norm=norm)
        # plt.show()
        plt.savefig("../plots/envs/board_"+self.config["name"]+".png")
        plt.clf()
        plt.close()
    
    
    def plot_transition(self, cur_state, action, next_state, path, name):
        _board = self.board.copy()
        
        
        for s in self.one_hot_mapping:
            
            if np.argmax(self.one_hot_mapping[s]) == np.argmax(cur_state):
                before = s
            
            if np.argmax(self.one_hot_mapping[s]) == np.argmax(next_state):
                after = s
                # print(self.one_hot_mapping[s], cur_state, next_state)
        
        _board[before] = 6
        _board[after] = 8
        act = self.actions[action]
        
        cmap = colors.ListedColormap(['black', 'grey', 'green', 'red', 'blue'])
        bounds = [0, 1, 3, 5, 7, 9]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        
        ax.imshow(_board, cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, 2.5, 1))
        ax.set_yticks(np.arange(-0.5, 2.5, 1))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.title("Action = "+act)
        
        # ax2 = fig.add_axes([0.75, 0.1, 0.02, 0.8])
        # cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
        #     spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
        # ax2.set_ylabel('Legend', size=12)
        
        
        plt.savefig(path+name+".png")
        plt.close()
        
    def abstract_state(self):
        '''
        Use current state and center to get higher state
        '''
        pass
    def extract_patch(self, center):
        '''
        Here I receive a pos and a patch around it is extracted
        around the patch
        '''
        
        patch_canvas = np.zeros(OBSERVATION_SIZE)
        pass
    
    def render_video(self):
        pass

    
    def states_to_onehot(self):
        pass