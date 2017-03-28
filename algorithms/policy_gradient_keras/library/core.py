from .config import *   # PGConfig
from .loss import *


def compute_gradients(agent):
    return tf.gradients(agent.loss.eval, agent.net.trainable_weights)


def apply_gradients(agent, optimizer_name):
    optimizer = None
    agent.gradients =\
        [tf.placeholder(v.dtype, v.get_shape()) for v in agent.net.trainable_weights]

    if optimizer_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=agent.lr)

    if optimizer is not None:
        return optimizer.apply_gradients(zip(agent.gradients, agent.net.trainable_weights))
