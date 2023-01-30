import math
import random as rd
import numpy as np
import gmpy2


def power_mode(b, e, m):
    return int(gmpy2.powmod(b, e, m))


def inverse_power_mode(x, m):
    return int(gmpy2.invert(x, m))


def generate_invertible_matrix(mtx_size):
    mtx = np.random.randint(low=0, high=50, size=(mtx_size, mtx_size))
    while np.linalg.det(mtx) == 0:
        mtx = np.random.randint(low=0, high=50, size=(mtx_size, mtx_size))
    invert_mtx = np.linalg.inv(mtx)

    return mtx, invert_mtx


def is_prime(mrc):
    maxDivisionsByTwo = 0
    ec = mrc - 1
    while ec % 2 == 0:
        ec >>= 1
        maxDivisionsByTwo += 1
    assert (2 ** maxDivisionsByTwo * ec == mrc - 1)

    def trial_composite(round_tester):
        if power_mode(round_tester, ec, mrc) == 1:
            return False
        for i in range(maxDivisionsByTwo):
            if power_mode(round_tester, 2 ** i * ec, mrc) == mrc - 1:
                return False
        return True

    numberOfRabinTrials = 20
    for i in range(numberOfRabinTrials):
        round_tester = rd.randrange(2, mrc)
        if trial_composite(round_tester):
            return False
    return True


def gen_prime(bit):
    x = rd.randint(2 ** (bit - 1) + 1, 2 ** bit - 1)
    for i in range(0, int(10 * math.log(x) + 3)):
        if is_prime(x):
            return x
        else:
            x += 1
    raise ValueError
