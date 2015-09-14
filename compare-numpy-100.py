#!/usr/bin/env python3
from datetime import datetime as dt
import itertools
from random import randint

import numpy100 as n100


def compare(functions, args_generator, gen_init=None, repeat=1000, comment="Default"):
    if gen_init is None:
        gen_init = []
    print(">" * 30)
    for f in functions:
        fname = f.__name__
        start = dt.now()
        for i, args in zip(range(repeat), args_generator(*gen_init)):
            f(*args)
        end = dt.now()
        total = (end - start).total_seconds()

        print("{comment}: {fname} run {repeat} times at {total}".format(**locals()))
    print("<" * 30)

def from_to_cycle(start=0, end=100):
    c = itertools.cycle(range(start, end))
    while 1:
        yield [next(c)]

def from_to_middle_cycle(start=1, end=101):
    c = from_to_cycle(start, end)
    while 1:
        n = next(c)[0]
        yield [n, n // 2]

def randint_gen(low=0, high=100):
    while 1:
        yield [randint(low, high)]

compare([n100.zeroes_py, n100.zeroes_np], from_to_cycle, comment="Small arrays")
compare([n100.zeroes_py, n100.zeroes_np], from_to_cycle, [10 ** 6, 10 ** 6 + 1000],
        comment="Large arrays")
compare([n100.zeroes_one_unit_py, n100.zeroes_one_unit_np], from_to_middle_cycle,
        comment="Small arrays")
compare([n100.zeroes_one_unit_py, n100.zeroes_one_unit_np], from_to_middle_cycle,
        [10 ** 6, 10 ** 6 + 1000], comment="Large arrays")
compare([n100.checkerboard_np, n100.checkerboard_py], randint_gen,
        comment="Small arrays", repeat=10)
compare([n100.checkerboard_np, n100.checkerboard_py], randint_gen, [10 ** 4, 10 ** 4 + 100],
        comment="Large arrays", repeat=1)
compare([n100.min_max_rand_matrix_np, n100.min_max_rand_matrix_py,
        n100.min_max_rand_matrix_py_alternative], randint_gen,
        comment="Small arrays", repeat=10)
compare([n100.min_max_rand_matrix_np, n100.min_max_rand_matrix_py,
        n100.min_max_rand_matrix_py_alternative], randint_gen, [10 ** 3, 10 ** 3 + 100],
        comment="Large arrays", repeat=10)
compare([n100.checkerboard_np, n100.checkerboard_tile_np], randint_gen, [10 ** 3, 10 ** 3 + 100],
        comment="Large arrays", repeat=10)
