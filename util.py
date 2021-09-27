import numpy as np
import math
from lime.wrappers.scikit_image import SegmentationAlgorithm

min = 1.0E-31
max = 1 - min

def log_odds(p):
    if p <= min:
        return math.log(min/max)
    elif p >= max:
        return math.log(max/min)
    else:
        return math.log(p/(1-p))
    
def change_in_log_odds(predict, original_sample, masked_sample, start, target, lime=False):
    if lime:
        original_sample = [original_sample]
        masked_sample = [masked_sample]
    original_pred_start = predict(original_sample)[0][start]
    masked_pred_start = predict(masked_sample)[0][start]
    original_pred_target = predict(original_sample)[0][target]
    masked_pred_target = predict(masked_sample)[0][target]
    
    change = log_odds(original_pred_start) - log_odds(masked_pred_start) + log_odds(masked_pred_target) - log_odds(original_pred_target)

    return change

def kl_divergence(p, q):
    """ Epsilon is used here to avoid conditional code for checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    p = p+epsilon
    q = q+epsilon
    divergence = np.sum(p*np.log(p/q))
    return divergence

def dim(a):
    if not type(a) == list:
        return 0
    return 1 + dim(a[0])

def anydup(thelist):
    seen = set()
    for x in thelist:
        if x in seen: return True
        seen.add(x)
    return False