import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

'''
Where the Laplacian returns a large number the field is rapidly going from not changing much at one point to changing a whole lot at another point. 
Where the Laplacian returns a small number the field is changing the same amount at nearby points. 
This "change in change" reflects the fact that the Laplacian is a second order differential operator, compared with the gradient, divergence, and curl, all of which are first order differential operators.
'''
A = np.matrix('1  2  3  4  1  1  2  1;'
              '2  2  3  0  1  2  2  1;'
              '3  0  38 39 37 36 3  0;'
              '4  1  40 44 41 42 2  1;'
              '1  2  43 44 40 39 1  3;'
              '2  0  39 41 42 40 2  0;'
              '1  2  0  2  2  3  1  1;'
              '0  2  1  3  1  0  4  2')
print(A)

L = np.matrix('0 -1 0;'
              '-1 4 -1;'
              '0 -1 0')
print(L)

grad = signal.convolve2d(A, L, boundary='fill', mode='valid')
print(grad)
# PLOT

fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
fig1 = plt.gcf()
ax_orig.imshow(A, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_mag.imshow(np.absolute(grad), cmap='gray')
ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

X, Y = np.meshgrid(np.arange(0, 6), np.arange(0, 6))
contour = ax_ang.contourf(X, Y, grad, 20, cmap='RdGy')

ax_ang.set_title('Gradient orientation')
ax_ang.set_axis_off()
plt.colorbar(contour, ax=ax_ang)
fig.show()
fig1.savefig('laplacian.pdf', dpi=150)
