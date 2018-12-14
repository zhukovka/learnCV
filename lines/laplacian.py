import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("test.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
box = cv2.boxFilter(img_gray, -1, (3, 3), normalize=False)
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
laplacian_box = cv2.Laplacian(box, cv2.CV_64F)

# converting back to uint8
abs_dst = cv2.convertScaleAbs(laplacian_box)

sobel = cv2.Sobel(abs_dst, cv2.CV_64F, 1, 1, ksize=5)
edges = cv2.Canny(abs_dst, 150, 250, apertureSize=3)
# sobel_abs = cv2.convertScaleAbs(sobel)
# threshold of the minimum number of intersections needed to detect a line
lines = cv2.HoughLines(img_gray, 1, np.pi / 180, 1)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.subplot(2, 2, 2), plt.imshow(box, cmap='gray')
plt.subplot(2, 2, 3), plt.imshow(abs_dst, cmap='gray')
plt.subplot(2, 2, 4), plt.imshow(edges, cmap='gray')
plt.show()
