import tensorflow as tf

from ..common import network


def make(config):
    kernel = "/cpu:0"
    if config.use_GPU:
        kernel = "/gpu:0"

    with tf.device(kernel):
        return network.make_shared_network(config)
