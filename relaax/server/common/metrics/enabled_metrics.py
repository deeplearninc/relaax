from __future__ import absolute_import

from . import metrics


class EnabledMetrics(metrics.Metrics):
    def __init__(self, options, metrics):
        self._options = options
        self._metrics = metrics

    def scalar(self, name, y, x=None):
        if self._enabled(name):
            self._metrics.scalar(name, y, x)

    def histogram(self, name, y, x=None):
        if self._enabled(name):
            self._metrics.histogram(name, y, x)

    def _enabled(self, name):
        return getattr(self._options, name, False)
