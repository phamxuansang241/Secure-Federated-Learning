import numpy as np


def generate_invertable_matrix(mtx_size):
    mtx = np.random.randint(low=0, high=50, size=(mtx_size, mtx_size))
    while np.linalg.det(mtx) == 0:
        mtx = np.random.randint(low=0, high=50, size=(mtx_size, mtx_size))
    invert_mtx = np.linalg.inv(mtx)

    return mtx, invert_mtx


def generate_mi_ni_matrix_client(mtx, invert_mtx, mtx_size, min_val, max_val):
    diagonal_vtc = np.random.randint(low=min_val, high=max_val-1, size=(mtx_size))
    mi_mtx = np.dot(np.dot(mtx, np.diag(diagonal_vtc)), invert_mtx)

    diagonal_vtc = np.random.randint(low=min_val, high=max_val-1, size=(mtx_size))
    ni_mtx = np.dot(np.dot(mtx, np.diag(diagonal_vtc)), invert_mtx)

    return mi_mtx, ni_mtx


def generate_matrix(mtx_size, min_val, max_val):
    mtx = np.random.randint(low=min_val, high=max_val-1, size=(mtx_size, mtx_size))
    return mtx



    
    
    
    




