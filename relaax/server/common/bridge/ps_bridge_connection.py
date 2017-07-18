from __future__ import absolute_import
from builtins import object
import grpc

from relaax.server.rlx_server.rlx_config import options
from relaax.server.common.metrics import metrics
from relaax.server.common.metrics import enabled_metrics

from . import bridge_pb2
from . import bridge_message

from relaax.common import profiling


profiler = profiling.get_profiler(__name__)


class PsBridgeConnection(object):
    def __init__(self, server):
        self._server = server
        self.session = PsBridgeSession(self)
        self.metrics = enabled_metrics.EnabledMetrics(options, PsBridgeMetrics(self))
        self._stub = None

    @property
    def stub(self):
        if self._stub is None:
            self._stub = bridge_pb2.BridgeStub(grpc.insecure_channel('%s:%d' % self._server))
            self._stub.Init(bridge_pb2.NullMessage())
        return self._stub


class PsBridgeSession(object):
    def __init__(self, connection):
        self.__connection = connection

    def __getattr__(self, name):
        return PsBridgeSessionMethod(self.__connection, [name])


class PsBridgeSessionMethod(object):
    def __init__(self, connection, names):
        self.connection = connection
        self.names = names

    def __getattr__(self, name):
        return PsBridgeSessionMethod(self.connection, self.names + [name])

    @profiler.wrap
    def __call__(self, *args, **kwargs):
        messages = bridge_message.BridgeMessage.serialize([self.names, list(args), kwargs])
        result = self.connection.stub.Run(messages)
        return bridge_message.BridgeMessage.deserialize(result)


class PsBridgeMetrics(metrics.Metrics):
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
