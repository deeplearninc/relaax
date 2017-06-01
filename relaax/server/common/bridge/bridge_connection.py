from __future__ import absolute_import
from builtins import object
import grpc

from relaax.server.common.metrics import metrics

from . import bridge_pb2
from . import bridge_message


class BridgeConnection(object):
    def __init__(self, server):
        self._server = server
        self.session = BridgeSession(self)
        self.metrics = BridgeMetrics(self)
        self._stub = None

    @property
    def stub(self):
        if self._stub is None:
            self._stub = bridge_pb2.BridgeStub(grpc.insecure_channel('%s:%d' % self._server))
            self._stub.Init(bridge_pb2.NullMessage())
        return self._stub


class BridgeSession(object):
    def __init__(self, connection):
        self.__connection = connection

    def __getattr__(self, name):
        return BridgeSessionMethod(self.__connection, [name])


class BridgeSessionMethod(object):
    def __init__(self, connection, names):
        self.connection = connection
        self.names = names

    def __getattr__(self, name):
        return BridgeSessionMethod(self.connection, self.names + [name])

    def __call__(self, *args, **kwargs):
        messages = bridge_message.BridgeMessage.serialize([self.names, list(args), kwargs])
        result = self.connection.stub.Run(messages)
        return bridge_message.BridgeMessage.deserialize(result)


class BridgeMetrics(metrics.Metrics):
    def __init__(self, connection):
        self.__connection = connection

    def scalar(self, name, y, x=None):
        kwargs = {'name': name, 'y': y}
        if x is not None:
            kwargs['x'] = bridge_pb2.ScalarMetric.Arg(value=x)
        self.__connection.stub.StoreScalarMetric(bridge_pb2.ScalarMetric(**kwargs))
