from __future__ import absolute_import

from . import metrics


class XMetrics(metrics.Metrics):
    def __init__(self, x, metrics):
        self._x = x
        self._metrics = metrics

    def scalar(self, name, y, x=None):
        if x is None:
            x = self._x()
        self._metrics.scalar(name, y, x)

    def histogram(self, name, y, x=None):
        if x is None:
            x = self._x()
        self._metrics.histogram(name, y, x)
