import tensorflow as tf

import relaax.algorithm_base.bridge_base
import relaax.algorithm_base.parameter_server_base
import relaax.server.common.saver.tensorflow_checkpoint

from . import network


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, config, saver_factory, metrics):
        self._network = network.make(config)

        self._session = tf.Session()

        self._saver = saver_factory(
            relaax.server.common.saver.tensorflow_checkpoint.TensorflowCheckpoint(
                self._session
            )
        )

        self._session.run(tf.variables_initializer(tf.global_variables()))

        self._bridge = _Bridge(metrics, self._network, self._session)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        checkpoint_ids = self._saver.checkpoint_ids()
        if len(checkpoint_ids) > 0:
            self._saver.restore_checkpoint(max(checkpoint_ids))

    def save_checkpoint(self):
        self._saver.save_checkpoint(self.global_t())

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
