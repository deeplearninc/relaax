from __future__ import print_function

import math
import concurrent
import grpc
import time
import signal
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
    def __init__(self, params):
        self._params = params

        kernel = "/cpu:0"
        if params.use_GPU:
            kernel = "/gpu:0"

        with tf.device(kernel):
            self._network = game_ac_network.make_shared_network(params, -1)\
                .apply_gradients(params)

        initialize = tf.initialize_all_variables()

        self._session = tf.Session()

        self._session.run(initialize)

        self._saver = tf.train.Saver()

    def service(self):
        return _Service(self)

    def close(self):
        self._session.close()

    def load_checkpoint(self, dir):
        checkpoint = tf.train.get_checkpoint_state(dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            return True
        return False

    def save_checkpoint(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self._saver.save(self._session, dir + '/' + 'checkpoint', global_step=self.global_t())

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
