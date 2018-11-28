import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
# Make data repeatable
np.random.seed(1981)

# Generate some random wells with random head (water table) observations
x, y, z = np.random.random((3, 10))

# Interpolate these onto a regular grid
xi, yi = np.mgrid[0:1:100j, 0:1:100j]
func = Rbf(x, y, z, function='linear')
zi = func(xi, yi)

# -- Plot --------------------------
fig, ax = plt.subplots()

# Plot flowlines
dy, dx = np.gradient(-zi.T) # Flow goes down gradient (thus -zi)
ax.streamplot(xi[:,0], yi[0,:], dx, dy, color='0.8', density=2)

# Contour gridded head observations
contours = ax.contour(xi, yi, zi, linewidths=2)
ax.clabel(contours)

# Plot well locations
ax.plot(x, y, 'ko')

plt.show()