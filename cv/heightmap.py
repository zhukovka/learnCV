import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

img_bgr = cv2.imread('zebra2.jpg')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# plt.imshow(img_gray, cmap='gray')
xx, yy = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
fig = plt.figure(figsize=(15, 15))
fig1 = plt.gcf()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, img_gray, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=2)
ax.view_init(85, 15)
plt.show()
fig1.savefig('zebra_heightmap1.pdf', dpi=150)
