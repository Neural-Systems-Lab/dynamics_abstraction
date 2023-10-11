import numpy as np

'''
Environment configurations
* Pixels
* Base Grids
* Compositional Grid
'''

# Pixel values
GOAL_PIXEL = 5
POS_PIXEL = 3
WALL_PIXEL = 0
EMPTY_PIXEL = 1

# Reward values
GOAL_REWARD = 10
STEP_REWARD = -1

# Composition

composite_config1 = {
    "id":1,
    "name":"composition1",
    "board":np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1]])

}

composite_config2 = {
    "id":1,
    "name":"composition2",
    "board":np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]])
}