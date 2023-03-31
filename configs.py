import pickle
import numpy as np

# TODO: These go into a config file later

# global variables

# Pixel inputs to the model
GOAL_PIXEL = 4
POS_PIXEL = 1
WALL_PIXEL = 2
EMPTY_PIXEL = 0
OBSERVATION_SIZE = (3, 3)
GOAL_REWARD = 10
STEP_REWARD = -0.1
OPTION_STEP_REWARD = -0.5

DENOMINATOR = 1
PRINT_ENV_PER_EPOCHS = 200

# Train Params: World Model
TOTAL_STEPS_DATA = 15000
TRAIN_EPOCHS_WM = 500
BATCH_SIZE_WM = 200
TRAIN_WM = False
LOAD_SAVED_WM = True

# Train Params: Agent
TRAIN_EPOCHS = 2500
TRAIN_EPOCHS_ABSTRACT_POLICY = 10
TRAIN_AGENT= False
LOAD_SAVED_AGENT = True

# Baseline PG agent
STEPS_PER_EPISODE = 25
BASELINE_TRAIN_EPOCHS = 2500
MAX_PLAN_STEPS = 1

# Save folder
SAVE_DIR = "/home/vsathish/core_projects/aistats/"


# Different environment configurations

config1 = {
    "id":1,
    "one_hot":[0, 1],
    "name": "3 * 3 Grid, Config 1",
    "rows": 3,
    "cols": 3,
    "goal_states": {(0, 0): [0, 0, 0, 0, 0, 0, 0, 1],   # A4
                    (2, 2): [0, 0, 0, 0, 0, 0, 1, 0],
                    (2, 0): [0, 0, 0, 0, 0, 1, 0, 0],
                    (0, 2): [0, 0, 0, 0, 1, 0, 0, 0]    # A3
                    },
    "walls": [(0, 1), (1, 1)],
    "higher_state": [(1, 1)],
}

config2 = {
    "id":2,
    "one_hot":[1, 0],
    "name": "3 * 3 Grid, Config 2",
    "rows": 3,
    "cols": 3,
    "goal_states": {(0, 0): [0, 0, 0, 1, 0, 0, 0, 0], 
                    (2, 2): [0, 0, 1, 0, 0, 0, 0, 0],   # A1
                    (2, 0): [0, 1, 0, 0, 0, 0, 0, 0],   # A7
                    (0, 2): [1, 0, 0, 0, 0, 0, 0, 0]    # A2
                    },
    "walls": [(1, 1), (2, 1)],
    "higher_state": [(1, 1)]
}

abstract_actions = [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0]
        ]


composition = np.array([
    [1, 0, 2, 0, 2],
    [0, 2, 0, 1, 0],
    [2, 0, 1, 0, 1]
])

# composition = np.array([
#     [1, 0, 2, 0, 1],
#     [0, 2, 0, 1, 0],
#     [2, 0, 1, 0, 1],
#     [0, 1, 0, 2, 0]
# ])


# Network configurations

lower_level_config = {
    "lr":1e-3,
    "b_lr":1e-3,
    "hyper_input_size": (8), # Can be batched
    "hypernet_layers": [16, 128, 128],
    "policy_input_size": (9), # Can be batched
    "policy_output_size": (4), # Can be batched
    "policy_layers": [64, 64], # [ip_sz*64, 64*op]
    "base_timesteps": 25,
    "action_logits_temperature":1,
    "l2_lambda":5e-3,
    "steps_threshold": 5, # Skip training if num_steps < threshold
    "action_mapping":{0:"up", 1:"down", 2:"left", 3:"right"}
}

higher_level_config = {
    "action_net_layers": [64, 128],
    "higher_input_size": 9,
    "higher_output_size": 4,
    "abstract_timestamps": 5,
    "lr": 1e-3
}


state_net_config = {
    "output_size":29
}

'''
One hot states
high dim states
REINFORCE without hypernet
Check if hypernet works => supervised/imitation learn

'''

# LARGE_SEQUENCE = []
# for a in range(4):
#     for b in range(4):
#         for c in range(4):
#             for d in range(4):
#                 for e in range(4):
#                     LARGE_SEQUENCE.append([a, b, c, d, e])

################################################
# Hypernet weight intialization techniques
################################################


# class hyperfanin_for_kernel(tf.keras.initializers.Initializer):
#     def __init__(self,fanin,varin=0.01,relu=True,bias=True):
#         self.fanin = fanin
#         self.varin = varin
#         self.relu = relu
#         self.bias = bias

#     def __call__(self, shape, dtype=None, **kwargs):
#         hfanin,_ = shape;
#         variance = (1/self.varin)*(1/self.fanin)*(1/hfanin)

#         if self.relu:
#             variance *= 2.0;
#         if self.bias:
#             variance /= 2.0;
        
#         variance = np.sqrt(3*variance);
#         # print("VARIANCE : ", variance)
#         return tf.random.uniform(shape, minval=-variance, maxval=variance)
#         #return tf.random.normal(shape)*variance
        
#     def get_config(self):  # To support serialization
#         return {"fanin": self.fanin, "varin": self.varin, "relu": self.relu, "bias": self.bias}
        


# class hyperfanin_for_bias(tf.keras.initializers.Initializer):
#     def __init__(self,varin=1.0,relu=True):
#         self.varin = varin
#         self.relu = relu

#     def __call__(self, shape, dtype=None, **kwargs):
#         hfanin,_ = shape;
#         variance = (1/2)*(1/self.varin)*(1/hfanin)
        
#         if self.relu:
#             variance *= 2.0;
        
#         variance = np.sqrt(3*variance);
        
#         return tf.random.uniform(shape, minval=-variance, maxval=variance)
#         #return tf.random.normal(shape)*variance

#     def get_config(self):  # To support serialization
#         return {"relu": self.relu, "varin": self.varin}
