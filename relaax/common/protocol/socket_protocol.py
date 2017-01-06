from __future__ import print_function

import base64
import io
import json
import logging
import numpy
import os
import random
import socket
import struct
import time


class Failure(Exception):
    def __init__(self, message):
        self.message = message


class AgentService(object):
    def act(self, state):
        raise NotImplementedError

    def reward_and_reset(self, reward):
        raise NotImplementedError

    def reward_and_act(self, reward, state):
        raise NotImplementedError


class AgentStub(AgentService):
    def __init__(self, socket):
        self._socket = socket

    def act(self, state):
        _sendf(self._socket, 'act', json.dumps(state, cls=_NDArrayEncoder))

    def reward_and_reset(self, reward):
        _sendf(self._socket, 'reward_and_reset', reward)

    def reward_and_act(self, reward, state):
        _sendf(self._socket, 'reward_and_act', reward, json.dumps(state, cls=_NDArrayEncoder))


def agent_dispatch(socket, agent_service):
    message = _receivef(socket)
    _debug('receive message %s', str(message)[:64])
    verb, args = message[0], message[1:]
    if verb == 'act':
        assert len(args) == 1
        agent_service.act(json.loads(args[0], object_hook=_ndarray_decoder))
        return
    if verb == 'reward_and_reset':
        assert len(args) == 1
        agent_service.reward_and_reset(args[0])
        return
    if verb == 'reward_and_act':
        assert len(args) == 2
        agent_service.reward_and_act(
            args[0],
            json.loads(args[1], object_hook=_ndarray_decoder)
        )
        return
    assert False


def environment_send_act(socket, action):
    _send(socket, 'act', action)


def environment_send_reset(socket, episode_score):
    _send(socket, 'reset', episode_score)


def environment_receive_act(socket):
    return _environment_receive(socket, 'act')


def environment_receive_reset(socket):
    return _environment_receive(socket, 'reset')


def _environment_receive(socket, verb):
    message = _receivef(socket)
    _debug('receive message %s', str(message)[:64])
    verb_, arg = message
    assert verb_ == verb
    return arg


def _socket_failure(socket_error):
    return Failure("socket error({}): {}".format(socket_error.errno, socket_error.strerror))


def _receivef(s):
    try:
        message = _receive(s)
    except socket.error as e:
        raise _socket_failure(e)
    if message is None:
        raise Failure('no message in socket')
    return message


def _sendf(s, *args):
    try:
        _send(s, *args)
    except socket.error as e:
        raise _socket_failure(e)


def _send(socket, *args):
    data = json.dumps(args, cls=_NDArrayEncoder)
    _debug('send data %s', data[:64])
    socket.sendall(''.join([
        struct.pack('<I', len(data)),
        data
    ]))


class _ReceiveError(Exception):
    pass
        

_COUNT_LEN = len(struct.pack('<I', 0))


def _receive(socket):
    try:
        return json.loads(
            _receiveb(
                socket,
                struct.unpack('<I', _receiveb(socket, _COUNT_LEN))[0]
            ),
            object_hook=_ndarray_decoder
        )
    except _ReceiveError:
        return None


def _receiveb(socket, length):
    packets = []
    rest = length
    while rest > 0:
        packet = socket.recv(rest)
        if not packet:
            raise _ReceiveError
        packets.append(packet)
        rest -= len(packet)
    data = ''.join(packets)
    assert len(data) == length
    return data


class _NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            output = io.BytesIO()
            numpy.savez_compressed(output, obj=obj)
            return {'b64npz': base64.b64encode(output.getvalue())}
        return json.JSONEncoder.default(self, obj)


def _ndarray_decoder(dct):
    """Decoder from base64 to numpy.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return numpy.load(output)['obj']
    return dct


def _debug(message, *args):
    logging.debug('%d:' + message, os.getpid(), *args)
