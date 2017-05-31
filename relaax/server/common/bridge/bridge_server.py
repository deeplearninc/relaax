from __future__ import absolute_import
from builtins import object
import concurrent
import grpc

from . import bridge_pb2
from . import bridge_message


class BridgeServer(object):
    def __init__(self, bind, init_ps):
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        bridge_pb2.add_BridgeServicer_to_server(Servicer(init_ps), self.server)
        self.server.add_insecure_port('%s:%d' % bind)

    def start(self):
        self.server.start()


class Servicer(bridge_pb2.BridgeServicer):
    def __init__(self, init_ps):
        self.init_ps = init_ps

    def Init(self, request, context):
        self.ps = self.init_ps()
        return bridge_pb2.NullMessage()

    def Run(self, request_iterator, context):
        names, args, kwargs = bridge_message.BridgeMessage.deserialize(request_iterator)
        last = self.ps.session
        for name in names:
            last = getattr(last, name)
        result = last(*args, **kwargs)
        return bridge_message.BridgeMessage.serialize(result)

    def StoreScalarMetric(self, request, context):
        x = None
        if request.HasField('x'):
            x = request.x.value
        self.ps.metrics.scalar(name=request.name, y=request.y, x=x)
        return bridge_pb2.NullMessage()
