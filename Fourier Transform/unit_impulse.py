import math


def unit_impulse(x):
    return 1 if x == 0 else 0


print(unit_impulse(42))
print(unit_impulse(0))
