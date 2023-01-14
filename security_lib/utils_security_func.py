import math
import random as rd
import numpy as np
import time
import gmpy2


def power_mode(b, e, m):
    res = 1
    while e > 0:
        if e & 1:
            res = (res*b) % m
        b = (b*b) % m
        e >>= 1

    return res


def generate_key(g, p, q, number_client):
    x = []
    X_arr = []
    y = []
    Y_arr = []
    R = []

    for i in range(number_client):
        xi = rd.randint(1, 1000)  # private key
        x.append(xi)
        Xi = power_mode(g, xi, p)  # public key
        X_arr.append(Xi)

        yi = rd.randint(1, 1000)  # private key
        y.append(yi)
        Yi = power_mode(g, yi, p)  # public key
        Y_arr.append(Yi)

    return x, X_arr, y, Y_arr


def compute_mul_public_key(p, X_arr, Y_arr, clients):
  X = 1
  Y = 1
  for client in clients:
      X = (X*X_arr[client.index]) % p
      Y = (Y*Y_arr[client.index]) % p
  return X, Y


def encrypt(g, p,
            x_cl, y_cl, X_mul, Y_mul,
            weight_vector, compress_nb):
    P_vector_cl = []
    weight_vector = weight_vector*compress_nb

    for i in range(len(weight_vector)):
        if weight_vector[i] < 0:
            g_v = int(gmpy2.powmod(g, -math.floor(weight_vector[i]), p))
            g_v = int(gmpy2.invert(g_v, p))
        else:
            g_v = int(gmpy2.powmod(g, math.floor(weight_vector[i]), p))

        Xy = int(gmpy2.powmod(X_mul, y_cl, p))
        numerator = (g_v * Xy) % p
        denominator = int(gmpy2.powmod(Y_mul, x_cl, p))
        denominator = int(gmpy2.invert(denominator, p))

        P_i_cl = (numerator * denominator) % p
        P_vector_cl.append(P_i_cl)
    return P_vector_cl


def decrypt(g, p, P_vector_list, clients, compress_nb):
    v = []
    nb_client = len(clients)
    # startTime = time.time()
    for i in range(len(P_vector_list[0])):
        K_i = 1
        for client in clients:
            K_i = (K_i * P_vector_list[client.index][i]) % p

        lower_bnd = -compress_nb*nb_client*2
        upper_bnd = compress_nb*nb_client*2

        for d in range(lower_bnd, upper_bnd + 1):
            if d < 0:
                gd = int(gmpy2.powmod(g, -d, p))
                gd = int(gmpy2.invert(gd, p))
            else:
                gd = int(gmpy2.powmod(g, d, p))

            if gd == K_i:
                # print('row {} column {}'.format(i, j))
                # print("solve = ", d)
                v.append(d)
                break
    # endTime = time.time()
    # print('Total time: {:.10f}s'.format(endTime - startTime))
    return v
