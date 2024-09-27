""" 
Written by: Hengchao Chen
Version: 0.1
Last modified date: 2024-09-02
Description: This file is used to define the module of hyperbolic spaces.
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import quad 

from scipy.optimize import root_scalar

# ---------------------- check dimension ---------------------- #

def check_dim(base, vector):

    # input base and vector should be either both 1D or both 2D

    if base.ndim == 1 and vector.ndim == 2:

        base = base.reshape(1, -1)

    elif base.ndim == 2 and vector.ndim == 1:

        vector = vector.reshape(1, -1)

    if base.shape[-1] != vector.shape[-1]:

        raise ValueError("The dimension of the base and the vector should be the same.")

    return base, vector

# ---------------------- Hyperbolic space ---------------------- #

# implement the hyperbolic space using the Hyperboloid model

# check Lectures on Hyperbolic Geometry for mathematical foundations

# visualize the hyperbolic space using the Poincare ball model

def minkowski_dot(v, w):

    v, w = check_dim(v, w)

    return - v[..., 0] * w[..., 0] + np.sum(v[..., 1:] * w[..., 1:], axis = -1)

def dist(base, target):

    # case 1: base is 1D and target is either 1D or 2D

    # case 2: base is 2D and target is 2D

    # the output shape is target.shape[:-1]

    base, target = check_dim(base, target)

    return np.arccosh(np.clip(- minkowski_dot(base, target), 1, None))

def exp(base, vector):

    # case 1: base is 1D and vector is either 1D or 2D

    # case 2: base is 2D and vector is 2D

    base, vector = check_dim(base, vector)

    vector_norm = np.sqrt(minkowski_dot(vector, vector))[..., np.newaxis]

    vector_norm_modified = np.clip(vector_norm, 1e-5, None)

    vector_unit = vector / vector_norm_modified

    return np.cosh(vector_norm) * base + np.sinh(vector_norm) * vector_unit

def log(base, target):

    # case 1: base is 1D and target is either 1D or 2D

    # case 2: base is 2D and target is 2D

    base, target = check_dim(base, target)

    dist_base_target = dist(base, target)[..., np.newaxis]

    sinh_dist_base_target = np.clip(np.sinh(dist_base_target), 1e-10, None)

    return dist_base_target * (target - np.cosh(dist_base_target) * base) / sinh_dist_base_target

# ---------------------- Random ---------------------- #

# One can manually design other riemannian radial distributions. Here we provide two examples: Riemannian Gaussian distributions, and a distribution on a unit ball.

# We generate random samples on the hyperboloid. 

def random_riemannian_gaussian(base = None, n_samples = 1, sigma = 1):

    # base is 1D and the output shape is n_samples x base.shape[-1]

    base = base.reshape(1, -1)

    dim_embedded = base.shape[-1]

    directions = np.random.randn(n_samples, dim_embedded)

    directions_tangent = directions + minkowski_dot(base, directions)[..., np.newaxis] * base

    directions_tangent_norm = np.sqrt(minkowski_dot(directions_tangent, directions_tangent))

    directions_tangent = directions_tangent / directions_tangent_norm[..., np.newaxis]

    # generate random radius from a distribution proportional to e^{-r^2/2\sigma^2} * \sinh^{dim_intrinsic-1}(r), where dim_intrinsic = dim_embedded - 1

    # use np.vectorize to vectorize the functions
    
    random_U = np.random.rand(n_samples)

    def integral(x, u):

        return quad(lambda t: np.exp(- np.arcsinh(t) ** 2 / (2 * sigma)) * t ** (dim_embedded - 2) * np.sqrt(1 + t ** 2), 0, x)[0] - u

    roots = np.array([root_scalar(integral, args = (u,), bracket = [0, 10]).root for u in random_U])

    vectors = directions_tangent * np.arcsinh(roots)[..., np.newaxis]

    return exp(base, vectors)

def random_uniform(base = None, n_samples = 1, radius = 1):

    # base is 1D and the output shape is n_samples x base.shape[-1]

    base = base.reshape(1, -1)

    dim_embedded = base.shape[-1]

    directions = np.random.randn(n_samples, dim_embedded)

    directions_tangent = directions + minkowski_dot(base, directions)[..., np.newaxis] * base

    directions_tangent_norm = np.sqrt(minkowski_dot(directions_tangent, directions_tangent))

    directions_tangent = directions_tangent / directions_tangent_norm[..., np.newaxis]

    length = np.random.rand(n_samples) * radius

    vectors = directions_tangent * length[..., np.newaxis]

    return exp(base, vectors)


# ---------------------- Fr\'echet mean ---------------------- #

def frechet_mean(data, stepsize = 0.1, tol = 1e-6, max_iter = 100):

    # data is 2D and the output shape is 1D

    # output the Frechet mean of the data

    data_poincare = hyperboloid_to_poincare_ball(data)

    mean_poincare = np.mean(data_poincare, axis = 0)

    mean = poincare_ball_to_hyperboloid(mean_poincare)

    for _ in range(max_iter):

        minus_gradient = np.mean(log(mean, data), axis = 0)

        mean_new = exp(mean, stepsize * minus_gradient)

        if np.linalg.norm(mean - mean_new) < tol:

            return mean_new
        
        mean = mean_new

    return mean


# ---------------------- Visualization ---------------------- #

def hyperboloid_to_poincare_ball(data):

    return data[..., 1:] / (data[..., 0][..., np.newaxis] + 1)

def poincare_ball_to_hyperboloid(data):

    w = 2 / (1 - np.sum(data ** 2, axis = -1))

    z = (1 + np.sum(data ** 2, axis = -1)) / (1 - np.sum(data ** 2, axis = -1))

    return np.concatenate((z[..., np.newaxis], w[..., np.newaxis] * data), axis = -1)

def visualize(data, transform_to_poincare_ball = True):

    # suitable for H2 space and poincare disk model

    if transform_to_poincare_ball:

        data = hyperboloid_to_poincare_ball(data)

    _, ax = plt.subplots(figsize = (5, 5))

    circle = plt.Circle((0, 0), 1, fill = False, edgecolor = 'black', lw = 2)

    ax.add_patch(circle)

    ax.scatter(data[:, 0], data[:, 1], color = 'black', s = 40, label = 'Local states')

    ax.axis('equal')

    ax.set_xlim(-1.02, 1.02)
    
    ax.set_ylim(-1.02, 1.02)

    ax.axis('off') 
 
