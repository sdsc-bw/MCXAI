import numpy as np
import math

def kl_divergence(p, q):
    """ Epsilon is used here to avoid conditional code for checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    p = p+epsilon
    q = q+epsilon
    #print(p/q)
    divergence = np.sum(p*np.log(p/q))
    return divergence

def dim(a):
    if type(a) == np.ndarray:
        if a.shape == (a.shape[0],):
            return 0
    elif not type(a) == list:
        return 0
    return 1 + dim(a[0])