import tensorflow as tf

from .bridge import bridge

class ParameterServer(object):
    def __init__(self, config, network, saver, metrics):
        self._config = config
        self._network = network
        self._saver = saver
        self._metrics = metrics

        initialize = tf.initialize_all_variables()

        self._session = tf.Session()

        self._session.run(initialize)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        return self._saver.restore_latest_checkpoint(self._session)

    def save_checkpoint(self):
        self._saver.save_checkpoint(self._session, self.global_t())

    def checkpoint_place(self):
        return self._saver.place()

    def global_t(self):
        return self._session.run(self._network.global_t)

    def increment_global_t(self):
        return self._session.run(self._network.increment_global_t)

    def apply_gradients(self, gradients):
        feed_dict = {p: v for p, v in zip(self._network.gradients, gradients)}
        feed_dict[self._network.learning_rate_input] = self._anneal_learning_rate(
            self._session.run(self._network.global_t)
        )
        self._session.run(self._network.apply_gradients, feed_dict=feed_dict)

    def get_values(self):
        return self._session.run(self._network.values)

    def store_scalar_metric(self, name, y, x=None):
        self._metrics.scalar(name, y, x=x)

    def service(self):
        return _Service(self)

    def _anneal_learning_rate(self, global_time_step):
        factor = (self._config.max_global_step - global_time_step) / self._config.max_global_step
        learning_rate = self._config.INITIAL_LEARNING_RATE * factor
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


class _Service(bridge.ParameterServerService):
    def __init__(self, parameter_server):
        self.increment_global_t = parameter_server.increment_global_t
        self.apply_gradients = parameter_server.apply_gradients
        self.get_values = parameter_server.get_values
        self.store_scalar_metric = parameter_server.store_scalar_metric
