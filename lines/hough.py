import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Probabilistic Line Transform
# minLinLength: The minimum number of points that can form a line.
# Lines with less than this number of points are disregarded.
minLineLength = 20
# maxLineGap: The maximum gap between two points to be considered in the same line.
maxLineGap = 50
# linesP = cv2.HoughLinesP(edges, 1, deg * np.pi / 180, threshold, None, minLineLength, maxLineGap)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.subplot(2, 2, 2), plt.imshow(img, cmap='gray')
plt.show()
