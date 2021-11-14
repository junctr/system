import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

x_j = np.linspace(-100, 100, 1000)
y_j = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(x_j, y_j)
Z = 100 * (Y - X**2)**2 + (1-X)**2

fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

ax.contour(X, Y, Z, 100)

fig.savefig("fig_banana.png")