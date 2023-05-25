# import cv2
import io
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


from configs import *

class CompositionalEnvironment:
    def __init__(self, base_configs, abstract_map, abstract_goal, lower_goal):
        # Let's say the current base maps are 3 x 3 grids
        # And assume only 2 base are there for now
        self.config1, self.config2 = base_configs
        self.m, self.n = self.config1["rows"], self.config1["cols"]
        
        # Let's not compose maps of different sizes
        assert self.config1["rows"] == self.config2["rows"]
        assert self.config1["cols"] == self.config2["cols"]
        
        self.abstract_map = np.array(abstract_map)
        self.rows, self.cols = self.abstract_map.shape
        
        # Also make sure number of base maps are same
        # as the max index of abstract map
        assert len(base_configs) >= np.max(self.abstract_map)
        
        self.abstract_goal = abstract_goal
        self.lower_goal = lower_goal
        self.historic_data = []
        self.episode_data = []
        self.video_data = []
        self.actions = {0:"U", 1:"D", 2:"L", 3:"R"}
        self.abstract_actions = [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0]
        ]
        self.__utils()
        
    def __utils(self):
        
        self.base1 = self.construct_base_env(self.config1)
        self.base2 = self.construct_base_env(self.config2)
        

        # Finally construct the big compositional maze
        self.env, self.env_goal = self.construct_abstract_environment(\
                    [self.base1, self.base2], \
                    self.abstract_map)

        # Get state mappings given a board
        self.abstract_mapping = self.get_mapping(self.abstract_map)
        self.config1_mapping = self.get_mapping(self.base1)
        self.config2_mapping = self.get_mapping(self.base2)
        
        self.abstract_to_lower_mapping = self.map_abstract_to_lower()
        self.valid_start_states = self.compute_valid_starts()
        
        # print(self.abstract_mapping, self.config1_mapping)
        
        # print(self.env)
        # print(self.base1)
        # print(self.base2)
        print("Final goal : ", self.env_goal)
        self.reset()
    
    
    def reset(self, start_position=None):
        
        if len(self.episode_data) > 0:
            # self.historic_data.append(self.episode_data)
            self.episode_data = []
            self.video_data = []
        
        idx = np.random.choice(range(len(self.valid_start_states)))
        
        # Important vars to track current state
        self.current_position = self.valid_start_states[idx]
        self.abstract_pos = random.choice(self.abstract_to_lower_mapping[self.current_position])
        self.lower_pos = self.get_lower_pos()
        
        if self.abstract_map[self.abstract_pos] == 1:
            self.current_config = self.config1
            self.current_room = self.base1
            lower_data = self.config1_mapping[self.lower_pos]
        else:
            self.current_config = self.config2
            self.current_room = self.base2
            lower_data = self.config2_mapping[self.lower_pos]
        
        # print("Current position after reset : \n", self.current_position, self.abstract_pos, \
            # self.lower_pos, self.abstract_map[self.abstract_pos])
        
        return self.abstract_mapping[self.abstract_pos], \
                self.current_room, lower_data
    
    def step(self, action):
        # Assume action is 0, 1, 2, 3 => argmax(action)
        
        # self.save_frame()
        
        reward = STEP_REWARD
        done = 1
        next_state = self.next_position(action)
        
        # If done. Logic is messy oops
        if next_state == self.env_goal:
            reward = GOAL_REWARD
            done = 0

        # Save current data to episode history before updating
        self.episode_data.append([self.current_position, self.abstract_pos, \
                                self.lower_pos, self.current_room, action, reward, \
                                done, next_state])
        
        
        
        self.current_position = next_state
        abstract_pos = self.abstract_to_lower_mapping[self.current_position]
        
        # Check for empty list
        assert len(abstract_pos) > 0
        # print("Abstract pos : ", abstract_pos)
        if len(abstract_pos) > 1:
            random.shuffle(abstract_pos)
            for pos in abstract_pos:
                if pos != self.abstract_pos:
                    self.abstract_pos = pos
        else:
            self.abstract_pos = abstract_pos[0]
        
        self.lower_pos = self.get_lower_pos()
        
        if self.abstract_map[self.abstract_pos] == 1:
            self.current_config = self.config1
            self.current_room = self.base1
            lower_data = self.config1_mapping[self.lower_pos]
        else:
            self.current_config = self.config2
            self.current_room = self.base2
            lower_data = self.config2_mapping[self.lower_pos]
        
        # print("Action : ", action)
        # print("########### NEXT STATES ##########\n", self.current_position, self.abstract_pos, \
            # self.lower_pos, self.abstract_map[self.abstract_pos])

        return self.abstract_mapping[self.abstract_pos],\
                self.current_room, lower_data, reward, done
    
    def abstract_step(self, abstract_action):
        # Reverse process. Given a lower level transition, 
        # Figure out the changes in higher states
        # _actions = self.abstract_actions

        # new_pos = self.lower_pos
        # print("new pos before change : ", new_pos)
        # for key in self.current_config["goal_states"].keys():
        #     if self.config1["goal_states"][key] == abstract_action:
        #         new_pos = key
        
        # if new_pos == None:
        #     for key in self.config2["goal_states"].keys():
        #         if self.config2["goal_states"][key] == abstract_action:
        #             new_pos = key
        
        # print("New position : ", new_pos, abstract_action)
        
        new_pos = abstract_action
        # Find global position
        x = self.abstract_pos[0]
        y = self.abstract_pos[1]
        
        x_, y_ = x*(self.m-1), y*(self.n-1)
        
        new_global_pos = (x_ + new_pos[0], y_ + new_pos[1])

        # print("new global pos : ", new_global_pos)
        reward = STEP_REWARD
        done = 1
        if new_global_pos == self.env_goal:
            reward = GOAL_REWARD
            done = 0

        # Save current data to episode history before updating
        self.episode_data.append([self.current_position, self.abstract_pos, \
                                self.lower_pos, self.current_room, abstract_action, reward, \
                                done, new_global_pos])
        
        
        
        self.current_position = new_global_pos
        abstract_pos = self.abstract_to_lower_mapping[self.current_position]
        
        # Check for empty list
        assert len(abstract_pos) > 0
        # print("Abstract pos : ", abstract_pos)
        if len(abstract_pos) > 1:
            random.shuffle(abstract_pos)
            for pos in abstract_pos:
                if pos != self.abstract_pos:
                    self.abstract_pos = pos
        else:
            self.abstract_pos = abstract_pos[0]
        
        self.lower_pos = self.get_lower_pos()
        
        if self.abstract_map[self.abstract_pos] == 1:
            self.current_room = self.base1
            lower_data = self.config1_mapping[self.lower_pos]
        else:
            self.current_room = self.base2
            lower_data = self.config2_mapping[self.lower_pos]
        
        # print("Action : ", abstract_action)
        # print("########### NEXT STATES ##########\n", self.current_position, self.abstract_pos, \
            # self.lower_pos, self.abstract_map[self.abstract_pos])

        return self.abstract_mapping[self.abstract_pos], \
                self.current_room, lower_data, reward, done
        
    
    def next_position(self, action):
        # print(action)
        state = self.current_position
        rows, cols = self.env.shape
        
        if action == 0:
            nxtState = (state[0] - 1, state[1])
        elif action == 1:
            nxtState = (state[0] + 1, state[1])
        elif action == 2:
            nxtState = (state[0], state[1] - 1)
        elif action == 3:
            nxtState = (state[0], state[1] + 1)
        
        # if next state legal
        if (nxtState[0] >= 0) and (nxtState[0] <= (rows -1)):
            if (nxtState[1] >= 0) and (nxtState[1] <= (cols -1)):
                if self.env[nxtState] != WALL_PIXEL:
                    return nxtState
        
        return state  
        
    def map_abstract_to_lower(self):
        # For every global maze state, list
        # the possible higher states
        rows, cols = self.env.shape
        p, q = self.abstract_map.shape
        
        mapping = {}
        
        for i in range(rows):
            for j in range(cols):
                mapping[(i, j)] = []
        
        for x in range(p):
            for y in range(q):
                if self.abstract_map[x, y] != 0:
                    i_ = (self.m-1)*x
                    j_ = (self.n-1)*y

                    for i in range(i_, i_+self.m):
                        for j in range(j_, j_+self.n):
                            mapping[(i, j)].append((x, y))
                
        # print(mapping)
        return mapping
        
    def compute_valid_starts(self):
        starts = []
        rows, cols = self.env.shape
        # print("env shape : ", self.env.shape)
        for i in range(rows):
            for j in range(cols):
                if self.env[i, j] != WALL_PIXEL and \
                    self.env[i, j] != GOAL_PIXEL:
                    starts.append((i,j))

        return starts
        
    def get_mapping(self, board):
        
        mapping = {}
        m, n = board.shape
        counter = 0
        for i in range(m):
            for j in range(n):
                temp = np.zeros(m*n)
                temp[counter] = 1.0
                mapping[(i, j)] = temp
                counter += 1
        
        return mapping
        
    def get_lower_pos(self):
        x = self.abstract_pos[0]
        y = self.abstract_pos[1]
        
        x_, y_ = x*(self.m-1), y*(self.n-1)
        
        return (self.current_position[0]-x_, self.current_position[1]-y_)
        
    def construct_abstract_environment(self, base_list, abstract_map):
        # A bit tricky to construct composable mazes
        # First construct a big maze
        # Initial maze is full of walls.
        m = self.m
        n = self.n
        rows = self.rows
        cols = self.cols
        
        env = np.zeros([(m-1)*rows + 1, (n-1)*cols + 1])
        
        # print(env.shape, abstract_map)
        
        # Now parse the abstract map while making changes to the
        # Main environment. Empty pixels get replaced
        
        # empty_patches = [] 
        for row in range(rows):
            for col in range(cols):
                _patch = abstract_map[row, col]
                
                if _patch == 0:
                    # walls
                    patch = np.ones([m, n])
                    # print(patch)
                    # Find relative position in the main env
                    p = (m-1) * row
                    q = (n-1) * col            
                    
                    env[p:p+m, q:q+n] = np.logical_or(patch, env[p:p+m, q:q+n])       
        
        env = -env
        # print(env)

        for row in range(rows):
            for col in range(cols):
                _patch = abstract_map[row, col]
                if _patch > 0:
                    # Base envs
                    patch = base_list[_patch-1]
                    
                    # Find relative position in the main env
                    p = (m-1) * row
                    q = (n-1) * col
                    
                    # Add the new patch. Wall gets priority
                    for i in range(m):
                        for j in range(n):
                            if patch[i, j] == 0 and env[p+i, q+j] == -1:
                                env[p+i, q+j] = 0
                            else:
                                env[p+i, q+j] = patch[i, j] or env[p+i, q+j]
                            
                    # env[p:p+m, q:q+n] = np.logical_or(patch, env[p:p+m, q:q+n])
        
        # print(env)
        
        _r, _c = env.shape
        for i in range(_r):
            for j in range(_c):
                if env[i, j] < 0:
                    env[i, j] = 1
                if env[i, j] == 0:
                    env[i, j] = 0
        
        
        
        # Find the global goal position
        p = (m-1)*self.abstract_goal[0]
        q = (n-1)*self.abstract_goal[1]
        p += self.lower_goal[0]
        q += self.lower_goal[1]
        
        # An inportant variable
        env_goal = (p, q)
        env[p, q] = GOAL_PIXEL
        
        
        print(env)
        return env, env_goal

    def construct_base_env(self, config):
        board = np.zeros([self.m, self.n])

        for x, y in config["walls"]:
            board[x, y] = WALL_PIXEL
            
        return board

    def plot_env(self, name):
        _env = np.copy(self.env)
        a, b = self.current_position
        a_, b_ = self.abstract_pos
        a_, b_ = a_ * (self.m-1), b_ * (self.n-1)
        
        cmap = ListedColormap(['black', 'gray', 'lightgreen'])
        plt.title("Compositional Environment")
        plt.imshow(self.env, cmap=cmap)
        
        plt.plot(b, a, 'ro', markersize=18)
        
        for data in self.episode_data:
            x, y = data[0]
            plt.plot(y, x, color=(1, 0, 0, 0.1), marker='o', markersize=8)
        
        x, y = self.episode_data[0][0]
        plt.plot(y, x, color=(0, 0, 1, 0.5), marker='o', markersize=14)
        
        rect = patches.Rectangle((b_-0.5, a_-0.5), self.m, self.n, \
                linewidth=4, edgecolor='r', facecolor='none')
        ax =  plt.gca()
        # ax.add_patch(rect)
        ax.grid(color='gray', linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(-0.5, 10.5, 1))
        ax.set_yticks(np.arange(-0.5, 7.5, 1))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # plt.axis('off')
        # plt.show()
        plt.savefig(name+".png", bbox_inches='tight')
        plt.close()
        
    def save_frame(self):
        # Same as plot
        a, b = self.current_position
        cmap = ListedColormap(['black', 'gray', 'lightgreen'])
        fig = plt.figure()
        plt.title("Compositional Environment")
        plt.imshow(self.env, cmap=cmap)
        plt.plot(b, a, 'ro', markersize=18)
        ax = plt.gca()
        ax.grid(color='gray', linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(-0.5, 6.5, 1))
        ax.set_yticks(np.arange(-0.5, 4.5, 1))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        
        # save as numpy array
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        # print(io_buf.getvalue())
        img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        img = cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)
        print(img, img.shape)
        self.video_data.append(img)
        plt.close()
        
    def render_video(self):
        size = 480, 640
        # duration = 4
        fps = 24
        out = cv2.VideoWriter(SAVE_DIR+'output.mp4', cv2.VideoWriter_fourcc(*"MJPG"), fps, (size[1], size[0]), False)
        for data in self.video_data:
            print("here : ", data.shape)
            
            out.write(data)

        out.release()

if __name__ == "__main__":

    base_configs = [config1, config2]
    # composition = [
    #     [0, 2, 2],
    #     [2, 0, 1],
    #     [2, 1, 2]
    # ]
    abstract_goal = (0, 2)
    lower_goal = (0, 2)
    my_env = CompositionalEnvironment(base_configs, composition, \
                                        abstract_goal, lower_goal)
    # my_env.plot_env(SAVE_DIR+"env_AISTATS")
    sample_actions = [0, 1, 2, 1, 2, 3, 1, 2, 3, 0, 0, 1, 2, 3, 0, 2, 3]
    
    for act in sample_actions:
        abstract, lower, reward, done = my_env.step(act)
        print("Abstract : ", abstract)
        print("Lower : ", lower)
        if done == 0:
            break

    # my_env.render_video()
    my_env.plot_env(SAVE_DIR+"env_AISTATS")
    my_env.reset()