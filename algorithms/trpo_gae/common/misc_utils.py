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


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor arguments.
    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code,
    such as MuJoCo and Atari from OpenAI Gym.
    """
    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args" : self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)
