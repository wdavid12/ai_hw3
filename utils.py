import numpy as np

def euclidean_distance(a,b):
    diff = a-b
    diff_squared = diff*diff
    if len(diff_squared.shape) > 1:
        return np.sqrt(diff_squared.sum(axis=1))
    else:
        return np.sqrt(diff_squared.sum())

