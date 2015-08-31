import math
import numpy
from timeit import timeit
from random import random


def sin_cos_math(x):
    return math.sin(x), math.cos(x)


def sin_cos_numpy(x):
    return numpy.sin(x), numpy.cos(x)

a30 = 30 * math.pi / 180
ar = random()

print("Math")
print(timeit("sin_cos_math(a30)", "from __main__ import sin_cos_math, a30"))
print(timeit("sin_cos_math(random())", "from __main__ import sin_cos_math;from random import random"))
print("Numpy")
print(timeit("sin_cos_numpy(a30)", "from __main__ import sin_cos_numpy, a30"))
print(timeit("sin_cos_numpy(random())", "from __main__ import sin_cos_numpy;from random import random"))
