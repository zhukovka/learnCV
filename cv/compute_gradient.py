import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

# load an image
star = cv2.imread('star.png', 0)
# Convert to normalized floating point
star = cv2.normalize(star.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def grad_1d(v: np.ndarray) -> np.ndarray:
    g_x = np.zeros(len(v))
    for x in range(len(v) - 1):
        g_x[x] = v[x + 1] - v[x]
    return g_x


def grad():
    # compute gradient
    # ∇F(x,y) = [F(x+1,y)-F(x,y), F(x,y+1)-F(x,y)]
    rows, cols = star.shape
    # gradient for x direction
    # 2D grayscale	(row, col) (y, x)
    g_x = np.zeros(star.shape)
    g_y = np.zeros(star.shape)
    for x in range(cols - 1):
        for y in range(rows - 1):
            g_x[y][x] = star[y][x + 1] - star[y][x]
            g_y[y][x] = star[y + 1][x] - star[y][x]
    return [g_x, g_y]


def plot_grad(g_x, g_y):
    fig, subplots = plt.subplots(2, 2)
    # (ax, ay) = subplots
    # (ax_orig, ax_x, ax_y, ax_sum) = ax
    ax_orig = subplots[0][0]
    ax_sum = subplots[0][1]
    ax_x = subplots[1][0]
    ax_y = subplots[1][1]
    ax_orig.imshow(star, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_x.imshow(g_x, cmap='gray')
    ax_x.set_title('Gradient in x direction')
    ax_x.set_axis_off()
    ax_y.imshow(g_y, cmap='gray')
    ax_y.set_title('Gradient in y direction')
    ax_y.set_axis_off()
    ax_sum.imshow(g_y + g_x, cmap='gray')
    ax_sum.set_title('Gradient in both direction')
    ax_sum.set_axis_off()
    fig.savefig('grad.pdf', dpi=150)
    plt.show()


def magnitude(g_x: np.ndarray, g_y: np.ndarray) -> np.ndarray:
    """
    Computes magnitude from x,y gradients
    ||F|| = √x² + y²

    :param g_x: gradient in x direction
    :param g_y: gradient in y direction
    :return: 2d array of vector magnitudes
    """
    rows, cols = g_x.shape
    m = np.zeros(g_x.shape)
    for x in range(cols):
        for y in range(rows):
            m[y][x] = math.sqrt(g_x[y][x] ** 2 + g_y[y][x] ** 2)
    return m


def angles(g_x: np.ndarray, g_y: np.ndarray) -> np.ndarray:
    """
    Θ = atan(y/x)
    :param g_x:
    :param g_y:
    :return:
    """
    rows, cols = g_x.shape
    a = np.zeros(g_x.shape)
    for x in range(cols):
        for y in range(rows):
            a[y][x] = np.arctan2(g_x[y][x], g_y[y][x])
    return a


grad_x, grad_y = grad()
grad_xy = grad_x + grad_y

mag = magnitude(grad_x, grad_y)


# plt.imshow(mag, cmap='gray')
# plt.imsave('star_edges.png', mag, cmap='gray')
# plt.show()

def plot_directions(x: np.ndarray, y: np.ndarray):
    directions = angles(x, y)
    plt.imshow(directions, cmap='gray')
    #
    # X : 1D or 2D array, sequence, optional
    # The x coordinates of the arrow locations
    #
    # Y : 1D or 2D array, sequence, optional
    # The y coordinates of the arrow locations
    #
    # U : 1D or 2D array or masked array, sequence
    # The x components of the arrow vectors
    #
    # V : 1D or 2D array or masked array, sequence
    # The y components of the arrow vectors
    U = mag * np.cos(directions)
    V = mag * np.sin(directions)
    plt.quiver(U, V)
    # plt.imsave('star_directions.png', cmap='gray')
    plt.show()


def add_noise(img: object, mean: float = 0, sigma: float = 0) -> object:
    im = np.zeros(img.shape, np.float)
    cv2.randn(im, mean, sigma)  # create the random distribution
    return cv2.add(img, im)  # add the noise to the original image


def plot_noise(img: object):
    noise1 = add_noise(img, sigma=0.1)
    noise5 = add_noise(img, sigma=0.5)
    grad0 = grad_1d(img[20])
    grad1 = grad_1d(noise1[20])
    grad2 = grad_1d(noise5[20])

    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 2), plt.imshow(noise1, cmap='gray')
    plt.title('Noise sigma 0.1'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3), plt.imshow(noise5, cmap='gray')
    plt.title('Noise sigma 0.5'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 4), plt.plot(grad0)
    plt.title('Gradient no noise')

    plt.subplot(2, 3, 5), plt.plot(grad1)
    plt.title('Gradient noise 0.1')

    plt.subplot(2, 3, 6), plt.plot(grad2)
    plt.title('Gradient noise 0.5')

    plt.show()


# plot_noise(star)
plot_directions(grad_x, grad_y)
