import numpy as np

def min_max_scale(x):
    if np.max(x) == np.min(x):
        return x
    return (x - np.max(x)) / (np.max(x) - np.min(x))
def standard_scale(x):
    return (x - np.mean(x)) / max(np.std(x),1e-7)
def max_abs_scale(x):
    return x / max(np.abs(np.max(x),1e-7))

