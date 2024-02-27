import pickle
import numpy as np

# TODO: These go into a config file later

# global variables

# Pixel inputs to the model
GOAL_PIXEL = 6
POS_PIXEL = 1
WALL_PIXEL = 2
EMPTY_PIXEL = 0
OBSERVATION_SIZE = (3, 3)
GOAL_REWARD = 10
STEP_REWARD = -0.1
OPTION_STEP_REWARD = -0.5

ENV_SAVE_PATH = "/Users/vsathish/Documents/Quals/plots/base_envs/"

# Actions
action_map = {
    0:"up",
    1:"down",
    2:"left",
    3:"right"
}

c1 = {
    "id":1,
    "name": "Base5 Config 1",
    "rows": 5,
    "cols": 5,
    "walls": [  (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),

                (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
            ],
    "subgoals":[(2, 0), (2, 4)],
    "abs_actions":[
        [
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
    ],
    "teleports":[
        [2],
        [3]
    ]

}


c2 = {
    "id":2,
    "name": "Base5 Config 2",
    "rows": 5,
    "cols": 5,
    "walls": [  (0, 0),                 (0, 3), (0, 4),
                (1, 0),                 (1, 3), (1, 4),
                                        (2, 3), (2, 4),
                (3, 0),                 (3, 3), (3, 4),
                (4, 0),                 (4, 3), (4, 4)
    ],
    "subgoals":[(2, 0), (4, 2), (0, 2)],
    "abs_actions":[
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ],
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ],
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]
    ],
    "teleports":[
        [2],
        [1],
        [0]

    ]
}


c3 = {
    "id":3,
    "name": "Base5 Config 3",
    "rows": 5,
    "cols": 5,
    "walls": [  (0, 0),(0, 1),       (0, 3), (0, 4),
                (1, 0),(1, 1),       (1, 3), (1, 4),
                                    
                (3, 0),(3, 1),       (3, 3), (3, 4),
                (4, 0),(4, 1),       (4, 3), (4, 4)
    ],
    "subgoals":[(2, 0), (4, 2), (0, 2)],
    "abs_actions":[
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ],
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ],
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]
    ],
    "teleports":[
        [2],
        [1],
        [0]

    ]
}

c4 = {
    "id":4,
    "name": "Base5 Config 4",
    "rows": 5,
    "cols": 5,
    "walls": [  (0, 0),(0, 1),       (0, 4),
                (1, 0),(1, 1),       (1, 4),
                (2, 0),(2, 1),                   
                (3, 0),(3, 1),       (3, 4),
                (4, 0),(4, 1),       (4, 4)
    ],
    "subgoals":[(2, 0), (4, 2), (0, 2)],
    "abs_actions":[
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ],
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ],
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]
    ],
    "teleports":[
        [2],
        [1],
        [0]

    ]
}


composite_config1 = {
    "id":1,
    "name":"composition1",
    "num_blocks":2,
    "block_size":(5, 5),
    "centers": [[ 0.33013344,  0.35932747, -0.31570193,  0.1935255 ],
                [-0.02549605,  0.0482652,  -0.03164882,  0.02881731]],
    "board":np.array([
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [GOAL_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL]]),

}

composite_config2 = {
    "id":2,
    "name":"composition2",
    "num_blocks":3,
    "block_size":(5, 5),
    "centers": [[ 0.33013344,  0.35932747, -0.31570193,  0.1935255 ],
                [-0.02549605,  0.0482652,  -0.03164882,  0.02881731]],
    "board":np.array([
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [EMPTY_PIXEL, EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL,EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL],
    [WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, WALL_PIXEL, EMPTY_PIXEL, EMPTY_PIXEL, WALL_PIXEL, WALL_PIXEL]])
}