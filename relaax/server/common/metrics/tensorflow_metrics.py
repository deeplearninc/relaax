import tensorflow as tf

import metrics


class TensorflowMetrics(metrics.Metrics):
    def __init__(self, metrics_dir, x):
        self._summaries = {}
        self._graph = tf.Graph()
        self._writer = tf.summary.FileWriter(metrics_dir, self._graph)
        self._session = tf.Session(graph=self._graph)
        self._x = x

    def scalar(self, name, y, x=None):
        with self._graph.as_default():
            if name not in self._summaries:
                placeholder = tf.placeholder(tf.float64)
                self._summaries[name] = (
                    placeholder,
                    tf.summary.scalar(name, placeholder)
                )
            placeholder, summary = self._summaries[name]
            self._writer.add_summary(
                self._session.run(summary, feed_dict={placeholder: y}),
                global_step=x if x is not None else self._x()
            )

