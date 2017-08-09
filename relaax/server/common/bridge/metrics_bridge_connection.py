from __future__ import absolute_import
from builtins import object
import grpc

from relaax.server.common.metrics import metrics
from relaax.server.common.metrics import enabled_metrics

from . import bridge_pb2
from . import bridge_message

from relaax.common import profiling


profiler = profiling.get_profiler(__name__)


class MetricsBridgeConnection(object):
    def __init__(self, options):
        self._server = options.metrics_server
        self.metrics = enabled_metrics.EnabledMetrics(options, BridgeMetrics(self))
        self._stub = None

    def set_x(self, x):
        message = bridge_pb2.X(x=x)
        self.send('SetX', lambda: message)

    def send(self, method_name, message_factory):
        if self._stub is None:
            self._stub = bridge_pb2.BridgeStub(grpc.insecure_channel('%s:%d' % self._server))
            for _ in range(9):
                method = getattr(self._stub, method_name)
                message = message_factory()
                try:
                    return method(message)
                except grpc.RpcError as e:
                    pass
        return getattr(self._stub, method_name)(message_factory())


class BridgeMetrics(metrics.Metrics):
    def __init__(self, connection):
        self.connection = connection

    @profiler.wrap
    def scalar(self, name, y, x=None):
        self.send('scalar', name=name, y=y, x=x)

    @profiler.wrap
    def histogram(self, name, y, x=None):
        self.send('histogram', name=name, y=y, x=x)

    def send(self, method, **kwargs):
        data = dict(method=method, kwargs=kwargs)
        self.connection.send('StoreMetric', lambda: bridge_message.BridgeMessage.serialize(data))
