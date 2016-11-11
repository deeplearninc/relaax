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
import io
import os
import numpy

import algorithms.a3c.game_ac_network
import shared
import ps_pb2


class Elapsed(object):
    def __init__(self):
        self._start = time.time()

    def __call__(self, message):
        end = time.time()
        # print(message, end - self._start)
        self._start = end


class _Ps(ps_pb2.PsServicer):
    def __init__(self, params, network, session):
        self._params = params
        self._network = network
        self._session = session
        self._initial_learning_rate = _log_uniform(
            params.INITIAL_ALPHA_LOW,
            params.INITIAL_ALPHA_HIGH,
            params.INITIAL_ALPHA_LOG_RATE
        )
        self._e = Elapsed()

    def IncrementGlobalT(self, request, context):
        return ps_pb2.Step(n=long(self._session.run(self._network.increment_global_t)))

    def ApplyGradients(self, request, context):
        self._e('ApplyGradients')
        grads = [
            numpy.ndarray(
                shape=a.shape,
                dtype=numpy.dtype(a.dtype),
                buffer=a.data
            )
            for a in request.arrays
        ]
        self._e('numpy.load')

        cur_learning_rate = self._anneal_learning_rate(
            self._session.run(self._network.global_t)
        )

        feed_dict = {p: v for p, v in zip(self._network.gradients, grads)}
        feed_dict[self._network.learning_rate_input] = cur_learning_rate

        self._session.run(self._network.apply_gradients, feed_dict=feed_dict)
        self._e('apply_gradients')
        return ps_pb2.NullMessage()

    def GetValues(self, request, context):
        self._e('GetValues')
        values = self._session.run(self._network.values)
        self._e('values')
        response = ps_pb2.NdArrayList(arrays=[
            ps_pb2.NdArrayList.NdArray(
                dtype=str(a.dtype),
                shape=a.shape,
                data=a.tobytes()
            )
            for a in values
        ])
        self._e('numpy.save')
        return response

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
    checkpoint_dir = 'checkpoints/' + 'boxing' + '_a3c_' + \
                          lstm_str + str(params.threads_cnt) + 'threads'

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
    ps_pb2.add_PsServicer_to_server(_Ps(params, global_network, sess), server)

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
