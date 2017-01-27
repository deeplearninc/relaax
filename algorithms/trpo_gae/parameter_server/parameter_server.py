import tensorflow as tf
import numpy as np
from time import sleep

import relaax.algorithm_base.parameter_server_base

from . import network


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, config, saver, metrics):
        self.n_iter = tf.Variable(0)
        self.is_collect = True

        self.policy_net, self.value_net = network.make(config)

        initialize = tf.variables_initializer(tf.global_variables())
        self._session = tf.Session()

        self.policy, self.baseline = network.make_head(config, self.policy_net, self.value_net, self._session)
        trpo_updater = network.make_trpo(self.policy, config, self._session)

        self._session.run(initialize)
        self._bridge = _Bridge(config, metrics, self)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        pass
        #return self._saver.restore_latest_checkpoint(self._session)

    def save_checkpoint(self):
        pass
        #self._saver.save_checkpoint(self._session, self.global_t())

    def checkpoint_location(self):
        pass
        #return self._saver.location()

    def bridge(self):
        return self._bridge


class _Bridge(object):
    def __init__(self, config, metrics, ps):
        self._config = config
        self._metrics = metrics
        self._ps = ps

    def wait_for_iteration(self):
        while not self._ps.is_collect:
            sleep(1)
        return self._ps.n_iter

    def send_experience(self, n_iter, paths):
        if n_iter == self._ps.n_iter:
            raise NotImplemented
            # call later (every 100)
            # save 2 nets & paths
            # iter for experience
            # self._ps.trpo_updater(paths)

    def receive_weights(self, n_iter):
        assert n_iter == self._ps.n_iter    # check
        return np.concatenate(self._ps.policy_net.get_trainable_weights(),
                              self._ps.value_net.get_trainable_weights())

    def metrics(self):
        return self._metrics
