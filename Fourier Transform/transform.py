import numpy as np
import matplotlib.pyplot as plt
# cosines of different frequencies
from numpy.core.multiarray import ndarray

n = 4  # number of periods
t: ndarray = np.linspace(0, n, n * 90)  # time samples

# amplitude
# −|A| ≤ Acosx ≤ |A|
A = 2

# y = cos(ωx)
# ω (omega) - (angular) frequency is the number of cycles
#               the sinusoid completes over a 2π interval
omega = (2 * np.math.pi)

f1: ndarray = A * np.cos(1 * omega * t)
f2: ndarray = A * np.cos(2 * omega * t)
f3: ndarray = A * np.cos(3 * omega * t)
f4: ndarray = A * np.cos(4 * omega * t)

# plt.subplot(2, 2, 1)
# plt.plot(t, f1)
# plt.title('2cos(x)')
# plt.subplot(2, 2, 2)
# plt.plot(t, f2)
# plt.title('2cos(2x)')
# plt.subplot(2, 2, 3)
# plt.plot(t, f3)
# plt.title('2cos(3x)')
# plt.subplot(2, 2, 4)
# plt.plot(t, f4)
# plt.title('2cos(4x)')
plt.rc('text', usetex=True)
plt.plot(t, f1 + f2 + f3 + f4)
plt.title(r"$\displaystyle\sum_{\omega=1}^4 2cos(\omega x)$")
plt.show()
