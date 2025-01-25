""" 
Written by: Hengchao Chen
Version: 0.1
Last modified date: 2024-09-02
Description: This file is used to define the module of the space of symmetric positive definite matrices.
"""

import numpy as np

from scipy.linalg import expm, logm

# ---------------------- check dimension ---------------------- #

def check_dim(base, vector):

    if base.ndim == 2 and vector.ndim == 3:

        base = base.reshape(1, base.shape[0], base.shape[1])

    elif base.ndim == 3 and vector.ndim == 2:

        vector = vector.reshape(1, vector.shape[0], vector.shape[1])

    elif base.ndim == 2 and vector.ndim == 2:

        base = base.reshape(1, base.shape[0], base.shape[1])

        vector = vector.reshape(1, vector.shape[0], vector.shape[1])
    
    if base.shape[-1] != vector.shape[-1] or base.shape[-2] != vector.shape[-2]:

        raise ValueError("The dimension of the base and the vector are not matched.")

    return base, vector

# ---------------------- Symmetric positive definite matrices ---------------------- #

def dist(base, points):

    if base.shape == points.shape:
        pass
    elif base.shape == points.shape[1:]:
        base = np.tile(base, (points.shape[0],) + (1,) * base.ndim)
    else:
        raise ValueError("The shape of base and points should be the same or the same except the first dimension")

    vals, vecs = np.linalg.eigh(base)

    base_sqrt_inv = np.einsum("...ij,...j,...kj->...ik", vecs, 1 / np.sqrt(vals), vecs)
    inner = np.einsum("...ij,...jk,...kl->...il", base_sqrt_inv, points, base_sqrt_inv)

    inner_vals, _ = np.linalg.eigh(inner)
    return np.linalg.norm(np.log(inner_vals), axis = -1)

def exp(base, vector, epsilon = 1e-10):

    # input case 1 : 2D base and 3D vector or 2D vector

    # input case 2 : 3D base and 3D vector

    if base.ndim == 2 and vector.ndim == 2:

        one_point = True
    
    else:

        one_point = False

    base, vector = check_dim(base, vector)

    # ensure symmetry

    vector = (vector + vector.transpose(0, 2, 1)) / 2 

    eigvals, eigvecs = np.linalg.eigh(base)

    base_sqrt = np.einsum('nij,nj,nkj->nik', eigvecs, np.sqrt(eigvals), eigvecs)

    #eigvals = np.maximum(eigvals, epsilon)

    base_sqrt_inv = np.einsum('nij,nj,nkj->nik', eigvecs, 1 / np.sqrt(eigvals), eigvecs)

    inner_exp = np.einsum('nij,njk,bkl->nil', base_sqrt_inv, vector, base_sqrt_inv)

    exp_inner = np.array([expm(m) for m in inner_exp])

    result = np.einsum('nij,njk,nkl->nil', base_sqrt, exp_inner, base_sqrt)

    if one_point:

        return result[0]
    
    return result

def log(base, target, epsilon = 1e-10):

    # input case 1 : 2D base and 3D vector

    # input case 2 : 3D base and 3D vector

    base, target = check_dim(base, target)

    eigvals, eigvecs = np.linalg.eigh(base)

    base_sqrt = np.einsum('nij,nj,nkj->nik', eigvecs, np.sqrt(eigvals), eigvecs)

    #eigvals = np.maximum(eigvals, epsilon)

    base_sqrt_inv = np.einsum('nij,nj,nkj->nik', eigvecs, 1 / np.sqrt(eigvals), eigvecs)

    inner_log = np.einsum('nij,njk,nkl->nil', base_sqrt_inv, target, base_sqrt_inv)

    log_inner = np.array([logm(m) for m in inner_log])

    return np.einsum('nij,njk,nkl->nil', base_sqrt, log_inner, base_sqrt)


# ---------------------- Random ---------------------- #

def random(base = None, n_samples = 1, radius = 1):

    dim = base.shape[-1]

    vector = np.random.randn(n_samples, dim, dim) * radius

    vector = (vector + vector.transpose(0, 2, 1)) / 2

    return exp(base, vector)


# ---------------------- Frechet mean ---------------------- #

def frechet_mean(data, stepsize = 0.1, max_iter = 100, tol = 1e-6):

    # input 3D data

    # initialize using the Euclidean mean

    mean = np.mean(data, axis = 0)

    for _ in range(max_iter):

        minus_gradient = np.mean(log(mean, data), axis = 0)

        mean_new = exp(mean, stepsize * minus_gradient)

        if np.linalg.norm(mean - mean_new) < tol:

            return mean_new
        
        mean = mean_new

    return mean 

