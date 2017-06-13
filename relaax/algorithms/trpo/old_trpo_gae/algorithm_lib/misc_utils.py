import numpy as np


def IDENTITY(x):
    return x


class TensorFlowLazyFunction(object):
    def __init__(self, inputs, outputs, session):
        self._inputs = inputs
        self._outputs = outputs
        self.session = session

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg
        return self.session.run(self._outputs, feeds)


# ================================================================
# Math utilities
# ================================================================

def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y-ypred)/vary


def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y-ypred)/vary
    out[vary < 1e-10] = 0
    return out


# ================================================================
# Misc
# ================================================================

def flatten(arrs):
    return np.concatenate([arr.flat for arr in arrs])


def unflatten(vec, shapes):
    i=0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i+size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)
