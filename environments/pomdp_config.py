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
}



c2 = {
    "id":2,
    "name": "Base5 Config 2",
    "rows": 3,
    "cols": 3,
    "walls": [  (0, 0),                 (0, 3), (0, 4),
                (1, 0),                 (1, 3), (1, 4),
                                        (2, 3), (2, 4),
                (3, 0),                 (3, 3), (3, 4),
                (4, 0),                 (4, 3), (4, 4)
    ],
}