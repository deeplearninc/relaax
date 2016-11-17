from __future__ import print_function

import tensorflow as tf

import worker


class Trainer(object):
    def __init__(self, params, master, log_dir='.'):
        kernel = "/cpu:0"
        if params.use_GPU:
            kernel = "/gpu:0"

        self._worker = worker.Factory(
            params=params,
            master=master,
            local_device=kernel,
            get_session=lambda: self.sess,
            add_summary=lambda summary, step:
                self._summary_writer.add_summary(summary, step)
        )()

        variables = [(tf.is_variable_initialized(v), tf.initialize_variables([v])) for v in tf.all_variables()]

        # prepare session
        self.sess = tf.Session()

        self._summary_writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

        for initialized, initialize in variables:
            if not self.sess.run(initialized):
                self.sess.run(initialize)

    def act(self, state):
        return self._worker.act(state)

    def reward_and_act(self, reward, state):
        return self._worker.reward_and_act(reward, state)

    def reward_and_reset(self, reward):
        return self._worker.reward_and_reset(reward)
