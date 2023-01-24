import numpy as np


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
    vtc = np.reshape(mtx, -1)

    # multiply vector with generator point of the curve
    temp_point_vtc = vtc * curve.g
    
    # get list form of temp point
    res_point_vtc = []
    for i in range(mtx_size**2):
        res_point_vtc.append([temp_point_vtc[0][i].x, temp_point_vtc[0][i].y])

    res_point_mtx = np.array(res_point_vtc).reshape(mtx_size, mtx_size, -1)
    return res_point_mtx