from __future__ import absolute_import

from . import metrics


class MultiMetrics(metrics.Metrics):
    def __init__(self, metrics):
        self._metrics = metrics

    def summary(self, summary, x=None):
        for m in self._metrics:
            m.summary(summary, x)

    def scalar(self, name, y, x=None):
        for m in self._metrics:
            m.scalar(name, y, x)

    def histogram(self, name, y, x=None):
        for m in self._metrics:
            m.histogram(name, y, x)
