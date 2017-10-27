from __future__ import absolute_import
from builtins import object
import concurrent
import grpc

from . import bridge_pb2
from . import bridge_message


class PsBridgeServer(object):
    def __init__(self, bind, ps_factory):
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        bridge_pb2.add_BridgeServicer_to_server(Servicer(ps_factory), self.server)
        self.server.add_insecure_port('%s:%d' % bind)

    def start(self):
        self.server.start()


class Servicer(bridge_pb2.BridgeServicer):
    def __init__(self, ps_factory):
        self.ps_factory = ps_factory

    def Init(self, request, context):
        self.ps = self.ps_factory()
        return bridge_pb2.NullMessage()

    def Run(self, request_iterator, context):
        names, args, kwargs = bridge_message.BridgeMessage.deserialize(request_iterator)
        last = self.ps.session
        for name in names:
            last = getattr(last, name)
        result = last(*args, **kwargs)
        return bridge_message.BridgeMessage.serialize(result)

    def NStep(self, request, context):
        return bridge_pb2.Step(value=self.ps.n_step())

    def StoreMetric(self, request_iterator, context):
        data = bridge_message.BridgeMessage.deserialize(request_iterator)
        getattr(self.ps.metrics, data['method'])(**data['kwargs'])
        return bridge_pb2.NullMessage()
