import numpy as np
from fastecdsa.point import Point


def generate_invertible_matrix(mtx_size):
    mtx = np.random.randint(low=0, high=50, size=(mtx_size, mtx_size))
    while np.linalg.det(mtx) == 0:
        mtx = np.random.randint(low=0, high=50, size=(mtx_size, mtx_size))
    invert_mtx = np.linalg.inv(mtx)

    return mtx, invert_mtx


def generate_mi_ni_matrix_client(mtx, invert_mtx, mtx_size, min_val, max_val):
    diagonal_vtc = np.random.randint(low=min_val, high=max_val-1, size=(mtx_size,))
    mi_mtx = np.dot(np.dot(mtx, np.diag(diagonal_vtc)), invert_mtx)

    diagonal_vtc = np.random.randint(low=min_val, high=max_val-1, size=(mtx_size,))
    ni_mtx = np.dot(np.dot(mtx, np.diag(diagonal_vtc)), invert_mtx)

    return mi_mtx, ni_mtx


def generate_matrix(mtx_size, min_val, max_val):
    mtx = np.random.randint(low=min_val, high=max_val-1, size=(mtx_size, mtx_size))
    return mtx


def ecc_multiply_matrix_with_point(mtx, mtx_size, curve):
    """
    example input:
    
    array([[5, 5, 2],
           [6, 7, 3],
           [6, 7, 4]])
    
    example output:

    array([[[ 6,  6], [ 6,  6], [ 2, 10]],
           [[ 5,  8], [10, 15], [ 8,  3]],
           [[ 5,  8], [10, 15], [12,  1]]])
    """
    # transform matrix to vector
    vtc = mtx.flatten()
    # multiply vector with generator point of the curve
    temp_point_vtc = vtc * curve.G
    # get list form of temp point
    res_point_vtc = []
    for i in range(mtx_size**2):
        res_point_vtc.append([temp_point_vtc[i].x, temp_point_vtc[i].y])

    res_point_mtx = np.array(res_point_vtc).reshape(mtx_size, mtx_size, -1)

    # print(res_point_mtx)
    return res_point_mtx


def ecc_add_pointmatrix_with_pointmatrix(point_mtx_a, point_mtx_b, mtx_size, curve):
    """
    Adding a matrix of points (point_mtx_a) with a matrix of points (point_mtx_b)
    mtx_size: size of point_mtx_a and point_mtx_b
    """
    res_point_mtx = []

    for i in range(mtx_size):
        for j in range(mtx_size):
            point = Point(point_mtx_a[i, j, 0], point_mtx_a[i, j, 1], curve) \
                + Point(point_mtx_b[i, j, 0], point_mtx_b[i, j, 1], curve)
            if point.x == 0:
                print('A: ', point_mtx_a[i, j, 0], point_mtx_a[i, j, 1])
                print('B: ', point_mtx_b[i, j, 0], point_mtx_b[i, j, 1])
                print('C: ', point.x, point.y)
            res_point_mtx.append([point.x, point.y])

    res_point_mtx = np.array(res_point_mtx).reshape(mtx_size, mtx_size, -1)
    return res_point_mtx


def ecc_subtract_pointmatrix_with_pointmatrix(point_mtx_a, point_mtx_b, mtx_size, curve):
    """
    Subtracting a matrix of points (point_mtx_a) by a matrix of points (point_mtx_b)
    mtx_size: size of point_mtx_a and point_mtx_b
    """
    res_point_mtx = []

    for i in range(mtx_size):
        for j in range(mtx_size):
            point = Point(point_mtx_a[i, j, 0], point_mtx_a[i, j, 1], curve) \
                - Point(point_mtx_b[i, j, 0], point_mtx_b[i, j, 1], curve)
            res_point_mtx.append([point.x, point.y])

    res_point_mtx = np.array(res_point_mtx).reshape(mtx_size, mtx_size, -1)
    return res_point_mtx


def ecc_multiply_matrix_with_pointmatrix(mtx, point_mtx, mtx_size, curve):
    """
    Multiplying a matrix (mtx) with a matrix of point (point_mtx)
    """
    res_point_mtx = []
    for i in range(mtx_size):
        for j in range(mtx_size):
            point = mtx[i][j] * Point(point_mtx[i, j, 0], point_mtx[i, j, 1], curve)
            res_point_mtx.append([point.x, point.y])
    
    res_point_mtx = np.array(res_point_mtx).reshape(mtx_size, mtx_size, -1)
    return res_point_mtx

