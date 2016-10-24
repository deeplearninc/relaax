from __future__ import print_function
import tensorflow as tf
import threading
import signal
import math
import os

from a3c.game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c.a3c_training_thread import A3CTrainingThread
from a3c.rmsprop_applier import RMSPropApplier

from a3c.params import *
from keras.optimizers import RMSprop


class Trainer:
    def __init__(self):
        self.device = "/cpu:0"
        if USE_GPU:
            self.device = "/gpu:0"

        self.initial_learning_rate = None
        self.global_t = 0
        self.stop_requested = False
        self.training_threads = []

    @staticmethod
    def log_uniform(lo, hi, rate):
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        v = log_lo * (1 - rate) + log_hi * rate
        return math.exp(v)

    def train_function(self, parallel_index, sess, summary_writer, summary_op, score_input):

        training_thread = self.training_threads[parallel_index]

        while True:
            if self.stop_requested:
                break
            if self.global_t > MAX_TIME_STEP:
                break

            diff_global_t = training_thread.process(sess, self.global_t, summary_writer,
                                                    summary_op, score_input)
            self.global_t += diff_global_t

    def signal_handler(self):
        def handler(signal, frame):
            print('You pressed Ctrl+C!')
            self.stop_requested = True
        return handler

    def run(self):
        self.initial_learning_rate = self.log_uniform(INITIAL_ALPHA_LOW,
                                                      INITIAL_ALPHA_HIGH,
                                                      INITIAL_ALPHA_LOG_RATE)
        # prepare session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                allow_soft_placement=True))
        if USE_LSTM:
            global_network = GameACLSTMNetwork(ACTION_SIZE, sess, -1, self.device)
        else:
            global_network = GameACFFNetwork(ACTION_SIZE, sess, self.device)

        learning_rate_input = tf.placeholder("float")

        grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                      decay=RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=RMSP_EPSILON,
                                      clip_norm=GRAD_NORM_CLIP,
                                      device=self.device)

        for i in range(PARALLEL_SIZE):
            training_thread = A3CTrainingThread(i, sess, global_network,
                                                self.initial_learning_rate,
                                                learning_rate_input,
                                                grad_applier, MAX_TIME_STEP,
                                                device=self.device)
            self.training_threads.append(training_thread)

        init = tf.initialize_all_variables()
        sess.run(init)

        # summary for tensorboard
        score_input = tf.placeholder(tf.int32)
        tf.scalar_summary("score", score_input)

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(LOG_FILE, sess.graph)

        # restore for keras
        self.global_t = global_network.restore(CHECKPOINT_DIR)

        train_threads = []
        for i in range(PARALLEL_SIZE):
            train_threads.append(threading.Thread(target=self.train_function,
                                                  args=(i, sess, summary_writer, summary_op, score_input)))

        signal.signal(signal.SIGINT, self.signal_handler())

        for t in train_threads:
            t.start()

        print('Press Ctrl+C to stop')
        signal.pause()

        print('Now saving data. Please wait')

        for t in train_threads:
            t.join()

        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        global_network.save(self.global_t, CHECKPOINT_DIR)

if __name__ == "__main__":
    model = Trainer()
    model.run()
