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


class AgentService(object):
    def act(self, state):
        raise NotImplementedError

    def reward_and_reset(self, reward):
        raise NotImplementedError

    def reward_and_act(self, reward, state):
        raise NotImplementedError


class AgentServiceFactory(object):
    def __call__(self, environment_service):
        raise NotImplementedError


class EnvironmentService(object):
    def act(self, action):
        raise NotImplementedError

    def reset(self, episode_score):
        raise NotImplementedError


class EnvironmentServiceFactory(object):
    def __call__(self, agent_service):
        raise NotImplementedError


def run_environment(server_address, environment_service_factory):
    while True:
        s = socket.socket()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            _connectf(s, _parse_address(server_address))
            environment_service = environment_service_factory(_AgentStub(s))
            while True:
                message = _receivef(s)
                _debug('receive message %s', str(message)[:64])
                verb, args = message[0], message[1:]
                if verb == 'act':
                    environment_service.act(args[0])
                if verb == 'reset':
                    environment_service.reset(args[0])
        except _Failure as e:
            _warning('{} : {}'.format(server_address, e.message))
            delay = random.randint(1, 10)
            _info('waiting for %ds...', delay)
            time.sleep(delay)
        finally:
            s.close()


def run_agent(socket, address, agent_service_factory):
    agent_service = agent_service_factory(_EnvironmentStub(socket))

    try:
        while True:
            message = _receivef(socket)
            _debug('from %s message %s', address, str(message)[:64])
            verb, args = message[0], message[1:]
            if verb == 'act':
                agent_service.act(json.loads(args[0], object_hook=_ndarray_decoder))
            if verb == 'reward_and_reset':
                agent_service.reward_and_reset(args[0])
            if verb == 'reward_and_act':
                agent_service.reward_and_act(
                    args[0],
                    json.loads(args[1], object_hook=_ndarray_decoder)
                )
    except _Failure as e:
        _warning('{} : {}'.format(address, e.message))


class _Failure(Exception):
    def __init__(self, message):
        self.message = message


def _failure(etype, e):
    return _Failure("{}({}): {}".format(etype, e.errno, e.strerror))


def _socket_failure(socket_error):
    return _failure('socket error', socket_error)


def _connectf(s, server_address):
    try:
        s.connect(server_address)
    except socket.error as e:
        raise _socket_failure(e)


def _receivef(s):
    try:
        message = _receive(s)
    except socket.error as e:
        raise _socket_failure(e)
    if message is None:
        raise _Failure('no message in socket')
    return message


def _sendf(s, *args):
    try:
        _send(s, *args)
    except socket.error as e:
        raise _socket_failure(e)


def _parse_address(address):
    host, port = address.split(':')
    return host, int(port)


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


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)


class _AgentStub(AgentService):
    def __init__(self, socket):
        self._socket = socket

    def act(self, state):
        _sendf(self._socket, 'act', json.dumps(state, cls=_NDArrayEncoder))

    def reward_and_reset(self, reward):
        _sendf(self._socket, 'reward_and_reset', reward)

    def reward_and_act(self, reward, state):
        _sendf(self._socket, 'reward_and_act', reward, json.dumps(state, cls=_NDArrayEncoder))


class _EnvironmentStub(EnvironmentService):
    def __init__(self, socket):
        self._socket = socket

    def act(self, action):
        _send(self._socket, 'act', action)

    def reset(self, episode_score):
        _send(self._socket, 'reset', episode_score)
