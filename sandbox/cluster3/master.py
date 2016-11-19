from __future__ import print_function

import sys
sys.path.append('../../server')

import math
import concurrent
import grpc
import time
import signal
import tensorflow as tf
import algorithms.a3c.params
import os

import algorithms.a3c.game_ac_network
import algorithms.a3c.bridge


class _Service(algorithms.a3c.bridge.MasterService):
    def __init__(self, params, network, session):
        self._params = params
        self._network = network
        self._session = session
        self._initial_learning_rate = _log_uniform(
            params.INITIAL_ALPHA_LOW,
            params.INITIAL_ALPHA_HIGH,
            params.INITIAL_ALPHA_LOG_RATE
        )

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
        learning_rate = self._initial_learning_rate * \
                        (self._params.max_global_step - global_time_step) / self._params.max_global_step 
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


def main():

    signal.signal(signal.SIGINT, lambda _1, _2: sys.exit(0))

    params = algorithms.a3c.params.Params()

    kernel = "/cpu:0"
    if params.use_GPU:
        kernel = "/gpu:0"

    with tf.device(kernel):
        global_network = algorithms.a3c.game_ac_network.make_shared_network(params, -1)

    initialize = tf.initialize_all_variables()

    sess = tf.Session()

    sess.run(initialize)

    lstm_str = ''
    if params.use_LSTM:
        lstm_str = 'lstm_'
    checkpoint_dir = 'checkpoints/boxing_a3c_' + lstm_str + '1threads'

    # init or load checkpoint with saver
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)

    global_t = sess.run(global_network.global_t)

    def stop_server(_1, _2):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_t)
        sess.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    algorithms.a3c.bridge.add_service_to_server(_Service(params, global_network, sess), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    last_global_t = None
    while True:
        time.sleep(1)
        global_t = sess.run(global_network.global_t)
        if global_t != last_global_t:
            last_global_t = global_t
            print("global_t is %d" % global_t)


def _log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


if __name__ == '__main__':
    main()
