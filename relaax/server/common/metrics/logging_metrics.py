from __future__ import absolute_import

import logging

from . import metrics


logger = logging.getLogger(__name__)
# this logger does not inherit logging level from higher level loggers
logger.setLevel(logging.INFO)


class LoggingMetrics(metrics.Metrics):
    def __init__(self, x):
        self._x = x

    def summary(self, summary, x=None):
        pass

    def scalar(self, name, y, x=None):
        self.emit(name, y, x)

    def histogram(self, name, y, x=None):
        self.emit(name, y, x)

    def emit(self, name, y, x):
        logger.info('%s(%s) == %s', name,
                    repr(x if x is not None else self._x()), repr(y))
