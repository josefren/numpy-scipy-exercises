from datetime import datetime as dt
import itertools

import numpy100 as n100


def compare(functions, args_generator, gen_init=None, repeat=1000, comment="Default"):
    if gen_init is None:
        gen_init = []
    for f in functions:
        fname = f.__name__
        start = dt.now()
        for i, args in zip(range(repeat), args_generator(*gen_init)):
            f(*args)
        end = dt.now()
        total = (end - start).total_seconds()
        print("{comment}: {fname} run {repeat} times at {total}".format(**locals()))

def from_to_cycle(start=0, end=100):
    c = itertools.cycle(range(start, end))
    while 1:
        yield [next(c)]

def from_to_middle_cycle(start=1, end=101):
    c = from_to_cycle(start, end)
    while 1:
        n = next(c)[0]
        yield [n, n // 2]

compare([n100.zeroes_py, n100.zeroes_np], from_to_cycle, comment="Small arrays")
compare([n100.zeroes_py, n100.zeroes_np], from_to_cycle, [10 ** 6, 10 ** 6 + 1000],
        comment="Large arrays")
compare([n100.zeroes_one_unit_py, n100.zeroes_one_unit_np], from_to_middle_cycle,
        comment="Small arrays")
compare([n100.zeroes_one_unit_py, n100.zeroes_one_unit_np], from_to_middle_cycle,
        [10 ** 6, 10 ** 6 + 1000], comment="Large arrays")
