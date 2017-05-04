import concurrent
import grpc

import bridge_pb2
import bridge_message


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
        op, feed_dict = bridge_message.BridgeMessage.deserialize(request_iterator)
        result = getattr(self.ps.session, op)(**feed_dict)
        return bridge_message.BridgeMessage.serialize(result)

    def StoreScalarMetric(self, request, context):
        x = None
        if request.HasField('x'):
            x = request.x.value
        self.ps.metrics.scalar(name=request.name, y=request.y, x=x)
        return bridge_pb2.NullMessage()
