import numpy as np

def euclidean_distance(a,b):
    """Euclidean distance for numpy arrays.
    For a pair of vectors, return the scalar distance.
    For a vector and a matrix, return the distance of the vector
    from each row of the matrix (useful for KNN).
    For a pair of matrices, return the distance between each pair
    of rows.
    """
    diff = a-b
    diff_squared = diff*diff
    return np.sqrt(diff_squared.T.sum(axis=0))


def L1_distance(a,b):
    diff = a-b
    diff_abs = np.abs(diff)
    return diff_abs.T.sum(axis=0)


def L3_distance(a,b):
    diff = a-b
    diff = np.abs(diff)**3
    return np.cbrt(diff.T.sum(axis=0))
