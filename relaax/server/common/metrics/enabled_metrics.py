from __future__ import absolute_import

from . import metrics


class EnabledMetrics(metrics.Metrics):
    def __init__(self, options, metrics):
        self._default = options.get('relaax_metrics_server/enable_unknown_metrics', False)
        self._options = options.get('relaax_metrics_server/metrics', {})
        self._metrics = metrics

    def summary(self, summary, x=None):
        self._metrics.summary(summary, x)

    def scalar(self, name, y, x=None):
        if self._enabled(name):
            self._metrics.scalar(name, y, x)

    def histogram(self, name, y, x=None):
        if self._enabled(name):
            self._metrics.histogram(name, y, x)

    def _enabled(self, name):
        return getattr(self._options, name, self._default)
