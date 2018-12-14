import numpy as np
import cv2
from matplotlib import pyplot as plt


def detect_edges(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # If a pixel gradient value is below the lower threshold, then it is rejected.
    low_threshold = 150
    # If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge
    # Canny recommended a upper:lower ratio between 2:1 and 3:1.
    threshold2 = low_threshold * 2
    # If the pixel gradient is between the two thresholds, then it will be accepted only if
    # it is connected to a pixel that is above the upper threshold
    edges = cv2.Canny(img_gray, low_threshold, threshold2, apertureSize=3)
    return edges


def detect_frame(edges):
    deg = 90
    # threshold of the minimum number of intersections needed to detect a line
    # line can be detected by finding the number of intersections between curves.
    # The more curves intersecting means that the line represented by that intersection have more points.
    # In general, we can define a threshold of the minimum number of intersections needed to detect a line.
    threshold = 25
    lines = cv2.HoughLines(edges, 1, deg * np.pi / 180, threshold)
    return lines


def draw_frame(img, frame):
    if frame is not None:
        for line in frame:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)


def test(file):
    img = cv2.imread(file)
    edges = detect_edges(img)
    lines = detect_frame(edges)
    draw_frame(img, lines)
    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
    plt.show()


# test_img = cv2.imread("test.png")
# print(lines)
test("test.png")
