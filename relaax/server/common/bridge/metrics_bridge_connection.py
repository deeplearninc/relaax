from __future__ import absolute_import
from builtins import object
import grpc

from relaax.server.metrics_server.metrics_server_config import options
from relaax.server.common.metrics import metrics
from relaax.server.common.metrics import enabled_metrics

from . import bridge_pb2
from . import bridge_message

from relaax.common import profiling


profiler = profiling.get_profiler(__name__)


class MetricsBridgeConnection(object):
    def __init__(self, server):
        self._server = server
        self.metrics = enabled_metrics.EnabledMetrics(options, BridgeMetrics(self))
        self._stub = None

    def set_x(self, x):
        self.stub.SetX(bridge_pb2.X(x=x))


    @property
    def stub(self):
        if self._stub is None:
            self._stub = bridge_pb2.BridgeStub(grpc.insecure_channel('%s:%d' % self._server))
        return self._stub


class BridgeMetrics(metrics.Metrics):
    def __init__(self, connection):
        self.connection = connection

    @profiler.wrap
    def scalar(self, name, y, x=None):
        self.send('scalar', name, y, x)

    @profiler.wrap
    def histogram(self, name, y, x=None):
        self.send('histogram', name, y, x)

    def send(self, method, name, y, x):
        messages = bridge_message.BridgeMessage.serialize(
                dict(method=method, kwargs=dict(name=name, y=y, x=x)))
        self.connection.stub.StoreMetric(messages)
