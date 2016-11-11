from __future__ import print_function

import numpy

import master_pb2


class Service(object):
    def increment_global_t(self):
        raise NotImplementedError

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError


def add_service_to_server(service, server):
    master_pb2.add_MasterServicer_to_server(_MasterServicer(service), server)


class Stub(object):
    def __init__(self, channel):
        self._stub = master_pb2.MasterStub(channel)

    def increment_global_t(self):
        return self._stub.IncrementGlobalT(master_pb2.NullMessage()).n

    def apply_gradients(self, gradients):
        self._stub.ApplyGradients(_build_ndarrays_message(gradients))

    def get_values(self):
        return _parse_ndarrays_message(self._stub.GetValues(master_pb2.NullMessage()))


class _MasterServicer(master_pb2.MasterServicer):
    def __init__(self, service):
        self._service = service

    def IncrementGlobalT(self, request, context):
        return master_pb2.Step(n=long(self._service.increment_global_t()))

    def ApplyGradients(self, request, context):
        self._service.apply_gradients(_parse_ndarrays_message(request))
        return master_pb2.NullMessage()

    def GetValues(self, request, context):
        return _build_ndarrays_message(self._service.get_values())


def _build_ndarrays_message(arrays):
    return master_pb2.NdArrays(arrays=[
        master_pb2.NdArrays.NdArray(
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
