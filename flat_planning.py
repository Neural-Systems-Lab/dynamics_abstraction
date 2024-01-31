import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

from environments.composition import CompositionGrid
from models.apc_planner import FlatPlanner
from environments.pomdp_config import *

# device = torch.device("mps")
COMPOSITION_CONFIG = composite_config2
BASE_CONFIGS = [c1, c2]
BATCH_SIZE = 1
INFERENCE_TIMESTEPS = 2
MAX_TIMESTEPS = 20
PLANNING_HORIZON = 1

def reset_state(env, start_state=None, goal=None, times=50):
    env.reset(start=start_state, goal=goal)
    # If the agent is in one of the subgoal state, reset.
    if len(env.get_higher_token()) > 1:
        if times == 0:
            print("Failed after max tries")
            return -1
        return reset_state(env, start_state, goal, times-1)

    return 0


planner = FlatPlanner(PLANNING_HORIZON)
env = CompositionGrid(COMPOSITION_CONFIG)
START_STATE = (2, 0)
GOALS = [(2, 0), (2, 4), (2, 8), (0, 9), (4, 9), (0, 10), (4, 10)]

####### Planner #######

counters = []
for goal in GOALS:
    print("############## GOAL : ", goal, " ##############")
    # Reset to a non-subgoal state
    reset_state(env, START_STATE, goal)

    # Plan
    steps, counter = planner.plan(env)
    counters.append(counter)

    # for s in steps:
    #     print(s)
    # print(goal)

print(counters)
print(GOALS)