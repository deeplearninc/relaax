import numpy as np


def categorical_sample(prob_nk):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_nk = np.asarray(prob_nk)
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    return np.argmax(csprob_nk > np.random.rand(N, 1), axis=1)
