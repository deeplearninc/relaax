import grpc

from relaax.server.common.metrics import metrics

import bridge_pb2
import bridge_message


class BridgeConnection(object):
    def __init__(self, server):
        self.__stub = bridge_pb2.BridgeStub(grpc.insecure_channel('%s:%d' % server))
        self.__stub.Init(bridge_pb2.NullMessage())
        self.session = BridgeSession(self.__stub)
        self.metrics = BridgeMetrics(self.__stub)


class BridgeSession(object):
    def __init__(self, stub):
        self.__stub = stub

    def __getattr__(self, name):
        return BridgeSessionMethod(self.__stub, name)


class BridgeSessionMethod(object):
    def __init__(self, stub, op):
        self.stub = stub
        self.op = op

    def __call__(self, **kwargs):
        feed_dict = kwargs
        messages = bridge_message.BridgeMessage.serialize([self.op, feed_dict])
        result = self.stub.Run(messages)
        return bridge_message.BridgeMessage.deserialize(result)


class BridgeMetrics(metrics.Metrics):
    def __init__(self, stub):
        self.__stub = stub

    def scalar(self, name, y, x=None):
        kwargs = {'name': name, 'y': y}
        if x is not None:
            kwargs['x'] = bridge_pb2.ScalarMetric.Arg(value=x)
        self.__stub.StoreScalarMetric(bridge_pb2.ScalarMetric(**kwargs))
