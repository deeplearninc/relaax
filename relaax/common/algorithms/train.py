from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .rewards import discounted_reward


# train policy with accumulated states, rewards and actions
def train_policy(obj):
    return obj.sess.run(obj.nn.partial_gradients, feed_dict={
        obj.nn.state: obj.states,
        obj.nn.action: obj.actions,
        obj.nn.discounted_reward: discounted_reward(np.vstack(obj.rewards))})
