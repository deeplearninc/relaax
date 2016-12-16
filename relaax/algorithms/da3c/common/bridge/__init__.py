from __future__ import print_function

import concurrent
import grpc
import numpy

from . import bridge_pb2


class PsService(object):
    def increment_global_t(self):
        raise NotImplementedError

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError


def start_ps(address, service):
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    bridge_pb2.add_ParameterServerServicer_to_server(_Servicer(service), server)
    server.add_insecure_port(address)
    server.start()
    return server


class PsStub(PsService):
    def __init__(self, ps):
        self._stub = bridge_pb2.ParameterServerStub(grpc.insecure_channel(ps))

    def increment_global_t(self):
        return self._stub.IncrementGlobalT(bridge_pb2.NullMessage()).n

    def apply_gradients(self, gradients):
        self._stub.ApplyGradients(_build_ndarrays_message(gradients))

    def get_values(self):
        return _parse_ndarrays_message(self._stub.GetValues(bridge_pb2.NullMessage()))


class _Servicer(bridge_pb2.ParameterServerServicer):
    def __init__(self, service):
        self._service = service

    def IncrementGlobalT(self, request, context):
        return bridge_pb2.Step(n=long(self._service.increment_global_t()))

    def ApplyGradients(self, request, context):
        self._service.apply_gradients(_parse_ndarrays_message(request))
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
