import tensorflow as tf

import relaax.algorithm_base.bridge_base
import relaax.algorithm_base.parameter_server_base
from network import GlobalPolicyNN


def make_network(config):
    network = GlobalPolicyNN(config)
    return network.apply_gradients()


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, config, saver, metrics):
        self._network = make_network(config)
        self._saver = saver

        initialize = tf.variables_initializer(tf.global_variables())

        self._session = tf.Session()

        self._session.run(initialize)

        self._bridge = _Bridge(metrics, self._network, self._session)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        return self._saver.restore_latest_checkpoint(self._session)

    def save_checkpoint(self):
        self._saver.save_checkpoint(self._session, self.global_t())

    def global_t(self):
        return self._session.run(self._network.global_t)

    def bridge(self):
        return self._bridge


class _Bridge(relaax.algorithm_base.bridge_base.BridgeBase):
    def __init__(self, metrics, network, session):
        self._metrics = metrics
        self._network = network
        self._session = session

    def increment_global_t(self):
        return self._session.run(self._network.increment_global_t)

    def apply_gradients(self, gradients):
        feed_dict = {p: v for p, v in zip(self._network.gradients, gradients)}
        self._session.run(self._network.apply_gradients, feed_dict=feed_dict)

    def get_values(self):
        return self._session.run(self._network.values)

    def metrics(self):
        return self._metrics
