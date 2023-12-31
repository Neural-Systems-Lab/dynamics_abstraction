import io
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from environments.pomdp_config import *


class CompositionGrid():
    def __init__(self, config, goal=None):
        
        # Define global constants
        self.config = config
        self.walls = []
        self.valid_pos = []
        self.one_hots = {}
        self.higher_states = {}
        self.episode_data = []
        self.historic_data = []

        self.fx, self.fy = config["block_size"] # composition width & height
        self.num_blocks = config["num_blocks"]

        # Define global variables
        self.board = config["board"]
        self.rows, self.columns = self.board.shape
        # print(self.rows, self.columns)
        self.state = None
        self.goal = None

        # Find walls and valid positions in the grid
        self.walls = self.wall_finder()
        
        # Map each position to a reference framed one hot
        # self.one_hots, self.higher_states = self.map_pos_to_one_hot()
        self.higher_states = self.map_higher_states()

        print(self.higher_states)

    def wall_finder(self):

        for i in range(self.rows):
            for j in range(self.columns):
                if self.board[i][j] == EMPTY_PIXEL:
                    self.valid_pos.append((i, j))
                else:
                    self.walls.append((i, j))
        
        return self.walls

    def get_state(self):
        if self.state == None:
            print("Init Error. Please reset the board to use for the first time")
            return 0

        return self.state

    def get_higher_token(self):
        if self.state == None:
            print("Init Error. Please reset the board to use for the first time")
            return 0
        return self.higher_states[self.state]
    
    def map_higher_states(self):
        higher_ = []

        for i in range(self.num_blocks):
            one_hot = np.zeros((self.num_blocks))
            one_hot[i] = 1
            higher_.append(one_hot)
        
        pos_to_higher = {}

        print(self.rows, self.fx, self.columns, self.fy)
        higher_rows = int((self.rows-1)/(self.fx-1))
        higher_cols = int((self.columns-1)/(self.fy-1))

        print( "In mapping function : ", higher_rows, higher_cols)
        counter = 0 # Higher state counter
        for i in range(0, higher_rows):
            for j in range(0, higher_cols):
                
                startx = i * (self.fx - 1)
                starty = j * (self.fy - 1)
                for x in range(startx, startx+self.fx):
                    for y in range(starty, starty+self.fy):
                        one_hot = np.zeros((self.num_blocks))
                        one_hot[counter] = 1
                        if (x, y) not in pos_to_higher.keys():
                            pos_to_higher[(x, y)] = []
                        
                        pos_to_higher[(x, y)].append(one_hot)
                
                counter += 1

        return pos_to_higher

    def get_pomdp_state(self):
        # Extract a 3 x 3 patch around the current state
        # Return a flattened version of the patch
        # print(self.state)
        patch = np.zeros((3, 3))
        for i in range(0, 3):
            for j in range(0, 3):
                r, c = self.state[0]-1+i, self.state[1]-1+j
                if r < 0 or r >= self.rows or c < 0 or c >= self.columns:
                    patch[i][j] = np.random.choice((WALL_PIXEL, EMPTY_PIXEL))
                else:
                    patch[i][j] = self.board[(r, c)]
        return patch.flatten()

    def reset(self, start=None, goal=None):
        if len(self.episode_data) > 0:
            self.historic_data.append(self.episode_data)
            self.episode_data = []
        
        if start == None:
            start = random.choice(self.valid_pos)
        
        if goal != None:
            self.goal = goal

        self.state = start
        # return self.one_hots[self.state]
        return self.get_pomdp_state()

    def step(self, action):
        
        next_state = self.next_position(action)
        reward, end = self.reward_function(next_state)
        
        self.episode_data.append((self.state, action, next_state, reward))
        self.state = next_state
        
        # return self.one_hots[self.state].flatten(), reward, end
        return self.get_pomdp_state(), reward, end
        
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
            if (nxtState[1] >= 0) and (nxtState[1] <= (self.columns -1)):
                if nxtState not in self.walls:
                    return nxtState
        
        return self.state
    
    def reward_function(self, state):
        
        # Returns reward and termination condition
        # Any transition not to goal gets -1 step reward
        # I use termination key as mask hence interchanged
        if state == self.goal:
            return GOAL_REWARD, 0
        else:
            return STEP_REWARD, 1

    def map_pos_to_one_hot(self):
        
        # assert self.rows % self.fx == 0
        # assert self.columns % self.fy == 0
        
        higher_state_counter = 0
        for i in range(0, self.rows, self.fx):
            for j in range(0, self.columns, self.fy):
                counter = 0
                for x in range(self.fx):
                    for y in range(self.fy):
                        one_hot = np.zeros((self.fx*self.fy))
                        one_hot_ = np.zeros((self.num_blocks))
                        one_hot[counter] = 1
                        one_hot_[higher_state_counter] = 1
                        self.one_hots[(i+x, j+y)] = one_hot
                        self.higher_states[(i+x, j+y)] = one_hot_
                        counter += 1
                higher_state_counter += 1
        
        return self.one_hots, self.higher_states


    def plot_board(self, board=None, save="../plots/envs/", name="composition1"):
        plt.clf()
        plt.close()
        if board:
            _board = board
        else:
            _board = self.board.copy()
        cmap = colors.ListedColormap(['black', 'gray', 'red', 'green'])
        bounds=[0,1,3,5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(which='major', axis='both', linestyle='-', color='darkgray', linewidth=2)
        ax.set_xticks(np.arange(-0.5, 100.5, 1))
        ax.set_yticks(np.arange(-0.5, 100.5, 1))

        plt.title("Environment - "+str(self.config["id"]))
        ax.imshow(_board, cmap=cmap, norm=norm)

        plt.savefig(save+name+".png")
        plt.clf()
        plt.close()

    def print_episode(self, episode):
        for j in range(len(self.historic_data[episode])):
            print(self.historic_data[episode][j])

    def episodic_video(self):
        return
        gx, gy = self.goal
        _board[gx][gy] = GOAL_PIXEL
        count = 1
        for s, a, s_ in self.episode_data:
            x, y = s
            _board[x][y] = POS_PIXEL
            plt.title("Step - "+str(count))
            count += 1