import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec

x = np.linspace(0, 2 * np.pi, 201)
y = 100 * np.sin(2 * np.pi * x)
# gray = (y + 1) / 2
img = [y] * 100

# fig = plt.figure(figsize=(7, 9))
# gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])
# #  Varying density along a streamline
# ax0 = fig.add_subplot(gs[0, :])
# ax0.plot(x, y)
# ax0.set_title('Varying Density')
#
# ax2 = fig.add_subplot(gs[1, 0])
# ax2.imshow(img, cmap='gray')
#
# ax3 = fig.add_subplot(gs[1, 1])
plt.subplot(2, 1, 1), plt.plot(x, y)
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.subplot(2, 1, 2), plt.imshow(img, cmap='gray')
plt.show()
