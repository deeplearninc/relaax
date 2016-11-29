from __future__ import print_function

import concurrent
import grpc
import numpy

import bridge_pb2
import master


def start_server(address, service):
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    bridge_pb2.add_MasterServicer_to_server(_MasterServicer(service), server)
    server.add_insecure_port(address)
    server.start()
    return server


class MasterStub(master.Service):
    def __init__(self, master):
        self._stub = bridge_pb2.MasterStub(grpc.insecure_channel(master))

    def increment_global_t(self):
        return self._stub.IncrementGlobalT(bridge_pb2.NullMessage()).n

    def apply_gradients(self, loss, vars):
        self._stub.ApplyGradients(bridge_pb2.LossVars(loss, _build_ndarrays_message(vars)))

    def get_values(self):
        return _parse_ndarrays_message(self._stub.GetValues(bridge_pb2.NullMessage()))


class _MasterServicer(bridge_pb2.MasterServicer):
    def __init__(self, service):
        self._service = service

    def IncrementGlobalT(self, request, context):
        return bridge_pb2.Step(n=long(self._service.increment_global_t()))

    def ApplyGradients(self, request, context):
        self._service.apply_gradients(request.loss, _parse_ndarrays_message(request.vars))
        return bridge_pb2.NullMessage()

    def GetValues(self, request, context):
        return _build_ndarrays_message(self._service.get_values())


def _build_ndarrays_message(arrays):
    return bridge_pb2.NdArrays(arrays=[
        bridge_pb2.NdArrays.NdArray(
            dtype=str(a.dtype),
            shape=a.shape,
            data=a.tobytes()
        )
        for a in arrays
    ])


def _parse_ndarrays_message(message):
    return [
        numpy.ndarray(
            shape=a.shape,
            dtype=numpy.dtype(a.dtype),
            buffer=a.data
        )
        for a in message.arrays
    ]
