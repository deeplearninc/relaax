from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def choose_action(probabilities):
    values = np.cumsum(probabilities)
    r = np.random.rand() * values[-1]
    return np.searchsorted(values, r)
