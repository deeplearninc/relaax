from __future__ import print_function

import base64
import io
import json
import logging
import numpy
import os
import random
import signal
import socket
import struct
import time


def run_environment(server_address, environment):
    while True:
        try:
            s = socket.socket()
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    s.connect(_parse_address(server_address))
                except socket.error as e:
                    raise _Failure("socket error({}): {} for {}".format(e.errno, e.strerror, server_address))

                try:
                    _send(s, 'act', _dump_state(environment))
                except socket.error as e:
                    raise _Failure("socket error({}): {} for {}".format(e.errno, e.strerror, server_address))
                while True:
                    message = _receive(s)
                    if message is None:
                        raise _Failure("no message from agent {}".format(server_address))
                    _debug('receive message %s', str(message)[:64])
                    verb, args = message[0], message[1:]
                    if verb == 'act':
                        reward, reset = environment.act(args[0])
                        if reset:
                            try:
                                _send(s, 'reward_and_reset', reward)
                            except socket.error as e:
                                raise _Failure("socket error({}): {} for {}".format(e.errno, e.strerror, server_address))
                        else:
                            try:
                                _send(s, 'reward_and_act', reward, _dump_state(environment))
                            except socket.error as e:
                                raise _Failure("socket error({}): {} for {}".format(e.errno, e.strerror, server_address))
                    if verb == 'reset':
                        environment.reset(args[0])
                        try:
                            _send(s, 'act', _dump_state(environment))
                        except socket.error as e:
                            raise _Failure("socket error({}): {} for {}".format(e.errno, e.strerror, server_address))
            finally:
                s.close()
        except _Failure as e:
            _warning(str(e))
            delay = random.randint(1, 10)
            _info('waiting for %ds...', delay)
            time.sleep(delay)


def run_agents(bind_address, agent_factory):
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    socket_ = socket.socket()
    try:
        _info('listening %s', bind_address)

        socket_.bind(_parse_address(bind_address))
        socket_.listen(100)

        n_agent = 0
        while True:
            connection, address = socket_.accept()
            _debug('accepted %s from %s', str(connection), str(address))
            try:
                pid = os.fork()
                if pid == 0:
                    socket_.close()
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    run_agent(connection, address, agent_factory(n_agent))
                    connection.close()
                    break
            finally:
                connection.close()
            n_agent += 1
    finally:
        socket_.close()


def run_agent(socket, address, agent):
    while True:
        message = _receive(socket)
        if message is None:
            break
        _debug('from %s message %s', address, str(message)[:64])
        verb, args = message[0], message[1:]
        if verb == 'act':
            state = json.loads(args[0], object_hook=_ndarray_decoder)
            _send(socket, 'act', agent.act(state))
        if verb == 'reward_and_reset':
            score = agent.reward_and_reset(args[0])
            if score is None:
                break
            _send(socket, 'reset', score)
        if verb == 'reward_and_act':
            state = json.loads(args[1], object_hook=_ndarray_decoder)
            action = agent.reward_and_act(args[0], state)
            if action is None:
                break
            _send(socket, 'act', action)


class _Failure(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


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


def _dump_state(environment):
    return json.dumps(environment.get_state(), cls=_NDArrayEncoder)


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
