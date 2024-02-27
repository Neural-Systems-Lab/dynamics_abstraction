import pickle
import numpy as np

# TODO: These go into a config file later

# global variables

# Pixel inputs to the model
GOAL_PIXEL = 0
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
    "name": "Config 1",
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
    "name": "Config 2",
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


########################
# Configs
########################

c1 = {
    "id":1,
    "name": "Base Config 1",
    "rows": 3,
    "cols": 3,
    "walls": [(0, 1), (0, 2), (1, 1), (1, 2)],
}

c2 = {
    "id":2,
    "name": "Base Config 2",
    "rows": 3,
    "cols": 3,
    "walls": [(1, 0), (2, 0), (2, 1)],
}

c3 = {
    "id":3,
    "name": "Base Config 3",
    "rows": 3,
    "cols": 3,
    "walls": [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)],
}

c4 = {
    "id":4,
    "name": "Base Config 4",
    "rows": 3,
    "cols": 3,
    "walls": [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],
}


composition = [1, 2]
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


