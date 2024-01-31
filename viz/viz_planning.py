import os
import sys

import numpy as np
import random
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import matplotlib.lines as mlines


x, y = (2, 0)
points = [(2, 0), (2, 4), (2, 8), (0, 9), (4, 9), (0, 10), (4, 10)]

apc_plan = [1, 1, 2, 10, 10, 10, 10]
flat_plan = [1, 4, 8, 11, 11, 12, 12]

distances = [np.abs(x-i)+np.abs(y-j) for i, j in points]

print(distances)

plt.plot(distances, apc_plan, label="APC Planning Steps")
plt.plot(distances, flat_plan, label="Flat Planning Steps")
plt.legend()
plt.xlabel("Distance between start and goal")
plt.ylabel("Total planning steps taken to reach the goal")
plt.title("APC Planning vs Flat Planning")
plt.savefig("/mmfs1/gscratch/rao/vsathish/quals/plots/planning/plan_compare.png", dpi=200)

