import logging
import numpy as np
import tensorflow as tf

from relaax.common.python.config.loaded_config import options
from relaax.common.algorithms.decorators import SubGraph

log = logging.getLogger(__name__)


class Placeholders(SubGraph):
    def build(self, ops):
        return [
            tf.Placeholder(dtype=op.dtype, shape=op.get_shape())
            for op in ops
        ]


def discounted_reward(self, rewards, gamma):
    # take 1D float array of rewards and compute discounted reward
    rewards = np.vstack(rewards)
    discounted_reward = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_reward[t] = running_add
    # size the rewards to be unit normal
    # it helps control the gradient estimator variance
    discounted_reward = discounted_reward.astype(np.float64)
    discounted_reward -= np.mean(discounted_reward)
    discounted_reward /= np.std(discounted_reward) + 1e-20

    return discounted_reward


def choose_action(probabilities):
    values = np.cumsum(probabilities)
    r = np.random.rand() * values[-1]
    return np.searchsorted(values, r)


def assemble_and_show_graphs(agent, parameter_server):
    with tf.variable_scope('parameter_server'):
        parameter_server()
    with tf.variable_scope('agent'):
        agent()
    log_dir = options.get("agent/log_dir", "log/")
    log.info(('Writing TF summary to %s. '
              'Please use tensorboad to watch.') % log_dir)
    tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
