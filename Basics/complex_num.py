import math
from matplotlib import pyplot as plt


def polar(c):
    """
    C = R + Ij
    C = |C|(cosθ + j sin⁡θ)

    :param c: - complex number in form R + Ij
    :return:
    """
    _R = c.real
    _I = c.imag
    # |C| = √(R²  + I²)
    _C = math.hypot(_R, _I)
    # tan θ = (I/R)
    # θ = arctan(I/R)
    # The arctan function returns angles in the range [-pi>2, pi>2].
    # However, because I and R can be positive and negative independently,
    # we need to be able to obtain angles in the full range [-pi, pi].
    # This is accomplished simply by keeping track of the sign of I and R when computing θ.
    # For example, python provides the function math.atan2(Imag, Real)
    _theta = math.atan2(_I, _R)
    return _C, _theta  # use return instead of print.


u = 3 + 5j
r, theta = polar(u)

plt.subplot(2, 1, 1, projection='polar')
# plt.plot(4, 5, 'o-')
plt.polar(theta, r, marker='o')

plt.subplot(2, 1, 2)
plt.plot(3, 5, 'o')
plt.xlim((0, 8))
plt.ylim((0, 8))
plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.show()
