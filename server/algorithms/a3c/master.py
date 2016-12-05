from __future__ import print_function

import math
import concurrent
import tensorflow as tf
import os
import game_ac_network


class Service(object):
    def increment_global_t(self):
        raise NotImplementedError

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError


class Master(object):
    def __init__(self, params, saver):
        self._params = params
        self._saver = saver

        kernel = "/cpu:0"
        if params.use_GPU:
            kernel = "/gpu:0"

        with tf.device(kernel):
            self._network = game_ac_network.make_shared_network(params, -1)

        initialize = tf.initialize_all_variables()

        self._session = tf.Session()

        self._session.run(initialize)

    def service(self):
        return _Service(self)

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

    def _anneal_learning_rate(self, global_time_step):
        factor = (self._params.max_global_step - global_time_step) / self._params.max_global_step
        learning_rate = self._params.INITIAL_LEARNING_RATE * factor
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


class _Service(Service):
    def __init__(self, master):
        self.increment_global_t = master.increment_global_t
        self.apply_gradients = master.apply_gradients
        self.get_values = master.get_values


def _log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


if __name__ == '__main__':
    main()
