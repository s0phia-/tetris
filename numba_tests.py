import numpy as np
from numba import njit


@njit
def list_insert():
    features = np.array([1, 2, 3, 4, 5])
    features[1, 3] = (6, 9)
    return features

list_insert()
