from __future__ import print_function

import tensorflow as tf


class Metrics(object):
    def scalar(self, name, y, x=None):
        raise NotImplementedError


class TensorFlowMetrics(Metrics):
    def __init__(self, metrics_dir):
        self._summaries = {}
        self._graph = tf.Graph()
        self._writer = tf.train.SummaryWriter(metrics_dir, self._graph)
        self._session = tf.Session(graph=self._graph)

    def scalar(self, name, y, x=None):
        with self._graph.as_default():
            if name not in self._summaries:
                placeholder = tf.placeholder(tf.float64)
                self._summaries[name] = (
                    placeholder,
                    tf.scalar_summary(name, placeholder)
                )
            placeholder, summary = self._summaries[name]
            self._writer.add_summary(
                self._session.run(summary, feed_dict={placeholder: y}),
                global_step=x
            )
