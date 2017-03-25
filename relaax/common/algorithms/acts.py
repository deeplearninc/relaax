from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def choose_action(probabilities):
    values = np.cumsum(probabilities)
    r = np.random.rand() * values[-1]
    return np.searchsorted(values, r)


# run agent's policy and get action
def action_from_policy(obj, state):
    action_probabilities = obj.sess.run(
        obj.nn.policy, feed_dict={obj.nn.state: [state]})
    return action_probabilities
