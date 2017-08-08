from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from . import metrics


class TensorflowMetrics(metrics.Metrics):
    def __init__(self, metrics_dir, x):
        self._scalar_summaries = {}
        self._histogram_summaries = {}
        self._graph = tf.Graph()
        self._writer = tf.summary.FileWriter(metrics_dir, self._graph)
        self._session = tf.Session(graph=self._graph)
        self._x = x

    def summary(self, summary, x=None):
        self._writer.add_summary(summary, global_step=x if x is not None else self._x())

    def scalar(self, name, y, x=None):
        self.add_summary(name, np.asarray(y), x, self._scalar_summaries, tf.summary.scalar)

    def histogram(self, name, y, x=None):
        self.add_summary(name, np.asarray(y), x, self._histogram_summaries, tf.summary.histogram)

    def add_summary(self, name, y, x, summaries, new_summary):
        y = np.asarray(y)
        with self._graph.as_default():
            if name not in summaries:
                placeholder = tf.placeholder(y.dtype)
                summaries[name] = (placeholder, new_summary(name, placeholder))
            placeholder, summary = summaries[name]
            self.summary(self._session.run(summary, feed_dict={placeholder: y}), x)
