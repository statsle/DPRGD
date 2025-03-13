""" 
Written by: Hengchao Chen
Version: 0.1
Last modified date: 2025-03-13
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

def check_nsamples(n):
    """Check the type of n and return it as a tuple"""
    
    if isinstance(n, int):
        return (n,)
    elif isinstance(n, tuple):
        return n
    else:
        raise ValueError("n should be an integer or a tuple of integers")

# ---------------------- Hyperbolic space ---------------------- #

# implement the hyperbolic space using the Hyperboloid model
# visualize the hyperbolic space using the Poincare ball model

def minkowski_dot(v, w): 
    v, w = check_dim(v, w) 
    return - v[..., 0] * w[..., 0] + np.sum(v[..., 1:] * w[..., 1:], axis = -1)

def minkowski_norm(vector, keepdims = False):
    """Return the Minkowski norm of the vector on the hyperboloid.
    When restricted to the tangent space to x in hyperboloid, the Minkowski norm is non-negative.
    Note that only when the vector is in the tangent space to x, the norm is non-negative."""
    if np.any(minkowski_dot(vector, vector) < 0):
        raise ValueError("The input vector is not in the tangent space to the hyperboloid")
    return np.sqrt(np.clip(minkowski_dot(vector, vector), 0, None)) if not keepdims else np.sqrt(np.clip(minkowski_dot(vector, vector), 0, None))[..., np.newaxis]

def dist(base, target):
    """
    Compute the distance between base and target on the hyperboloid.
    The output shape is target.shape[:-1]""" 

    base, target = check_dim(base, target) 
    return np.arccosh(np.clip(- minkowski_dot(base, target), 1, None))

def exp(base, vector):
    base, vector = check_dim(base, vector) 
    vecnorm = np.sqrt(minkowski_dot(vector, vector))[..., np.newaxis]
    vecnorm_modified = np.clip(vecnorm, 1e-5, None)
    vecunit = vector / vecnorm_modified
    return np.cosh(vecnorm) * base + np.sinh(vecnorm) * vecunit

def log(base, target):
    base, target = check_dim(base, target) 
    dist_base_target = dist(base, target)[..., np.newaxis] 
    sinh_dist_base_target = np.clip(np.sinh(dist_base_target), 1e-10, None) 
    return dist_base_target * (target - np.cosh(dist_base_target) * base) / sinh_dist_base_target

# ---------------------- Random ---------------------- #

# Simulate Riemannian Gaussian distributions, and a distribution on a unit ball.

def random_vector(base):
    """Generate random tangent vectors on the hyperboloid to the base point(s)"""
    directions = np.random.randn(*base.shape)
    directions = directions + minkowski_dot(directions, base)[..., np.newaxis] * base
    return directions / minkowski_norm(directions, keepdims = True)

def random_radius(base, sigma, type = "rie_normal"):
    """Generate random radii on the hyperboloid according to the riemannian radial distributions. This function is used in sampler.
        
    - rie_normal : propto exp(-r^2/(2*sigma^2)) sinh^(dim-1)(r)
    - rie_laplace : propto exp(-r/sigma) sinh^(dim-1)(r)
        
    Use inverse function sampling to sample r in [0, inf). For approximation, we restrict r to [0,20] as we assume that the probability that r>20 is negligible.
    When dim is high, the quad(f, a, b) is not accurate due to the small values of f and the discrete nature of quad. 
    Therefore, this limits the use of quad in high-dimensional spherical data analysis.
    For current implementation, we restrict the dimension to be less than 6.
        
    Also, when using rie_laplace, we require the sigma to be small enough to ensure that the distribution is well defined."""

    if type not in ["rie_normal", "rie_laplace"]:
        raise ValueError("The type of distribution is not supported at this moment")
        
    if base.ndim <= 1:
        raise ValueError("The input base should have at least two dimensions")
        
    dim = base.shape[-1] - 1
    n = base.shape[:-1]
    n = check_nsamples(n)
    U = np.random.rand(*n)

    if type == "rie_normal":
        f = lambda r: np.exp(-r ** 2 / (2 * sigma ** 2) + np.log(np.sinh(r)) * (dim - 1)) # when r is large, sinh(r) is approximately exp(r)/2
    else:
        f = lambda r: np.exp(-r / sigma + np.log(np.sinh(r)) * (dim - 1)) # when r is large, sinh(r) is approximately exp(r)/2        
    c = quad(f, 0, 20)[0]
            
    def res(r, u):
        return quad(f, 0, r)[0] - u * c

    def find(u):
        if type == "rie_normal":
            return root_scalar(res, bracket = [1e-40, 20], args = (u,)).root
        else:
            return root_scalar(res, bracket = [1e-40, 20 / (1/sigma - dim)], args = (u,)).root # 1/sigma must be larger than dim - 1 to ensure that the distribution is well defined
                    
    return np.vectorize(find)(U)[..., np.newaxis]

def random_riemannian_gaussian(base = None, n_samples = None, sigma = None):
    """base is 1D and the output shape is n_samples x base.shape[-1]"""
    n = check_nsamples(n_samples)
    base = np.tile(base, n + (1,) * base.ndim) 
    vector = random_vector(base = base)
    radii = random_radius(base = base, sigma = sigma, type = 'rie_normal')
    return exp(base, radii * vector)

# ---------------------- Frechet mean ---------------------- #

def frechet_mean(data, stepsize = 0.1, tol = 1e-6, max_iter = 100):
    """
    Output the Frechet mean of the data on the hyperboloid.
    
    Input: 2D data""" 

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


 
  

    
 
     
     
    
   
    
    
         
 
