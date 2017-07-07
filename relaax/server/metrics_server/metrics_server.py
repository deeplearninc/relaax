from __future__ import absolute_import
from __future__ import division
from builtins import object
import logging
import os
import ruamel.yaml
import time
import threading
import signal

# Load configuration options
# do it as early as possible
from .metrics_server_config import options
from relaax.common import profiling
from relaax.server.common.bridge import metrics_bridge_server
from relaax.server.common.metrics import enabled_metrics
from relaax.server.common.metrics import logging_metrics
from relaax.server.common.metrics import multi_metrics
from relaax.server.common.metrics import tensorflow_metrics
import multiprocessing
try:
    from Queue import Empty  # noqa
except ImportError:
    from queue import Empty  # noqa


logger = logging.getLogger(__name__)


class MetricsServer(object):

    @classmethod
    def exit_server(cls, signum, frame):
        cls.stopped_server = True

    @classmethod
    def start(cls):
        try:
            profile_dir = options.get('metrics_server/profile_dir')
            if profile_dir is not None:
                profiling.set_handlers([profiling.FileHandler(os.path.join(
                                        profile_dir, 'metrics.txt'))])
                profiling.enable(True)

            logger.info("Starting metrics server on %s:%d" % options.bind)

            # keep the server or else GC will stop it
            server = metrics_bridge_server.MetricsBridgeServer(options.bind, MetricsHandler())
            server.start()

            events = multiprocessing.Queue()
            signal.signal(signal.SIGINT, cls.exit_server)
            signal.signal(signal.SIGTERM, cls.exit_server)
            cls.stopped_server = False

            while not cls.stopped_server:
                #time.sleep(1)
                try:
                    msg = events.get(timeout=1)
                except Empty:
                    pass
                except:
                    break

        except KeyboardInterrupt:
            # swallow KeyboardInterrupt
            pass
        except:
            raise

    @staticmethod
    def metrics_factory(x):
        metrics = []
        if options.get('metrics_server/log_metrics', False):
            metrics.append(logging_metrics.LoggingMetrics(x))
        metrics_dir = options.get('metrics_server/metrics_dir')
        if metrics_dir is not None:
            metrics.append(tensorflow_metrics.TensorflowMetrics(metrics_dir, x))
        return enabled_metrics.EnabledMetrics(options.get('metrics'),
                                              multi_metrics.MultiMetrics(metrics))


class MetricsHandler(object):
    def __init__(self):
        self._x = 0.
        self.metrics = MetricsServer.metrics_factory(lambda: self._x)

    def set_x(self, x):
        self._x = x


def main():
    MetricsServer.start()


if __name__ == '__main__':
    main()
