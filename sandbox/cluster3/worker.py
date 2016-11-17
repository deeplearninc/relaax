from __future__ import print_function

import sys
sys.path.append('../../server')

import base64
import flask
import flask_socketio
import grpc
import io
import json
import logging
import numpy
import os
import signal
import socket
import struct

import algorithms.a3c.params
import algorithms.a3c.master
import algorithms.a3c.worker

def info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def handle_connection(s, address, n_worker):
    info('%s: start', address)
    params = algorithms.a3c.params.Params()

    lstm_str = ''
    if params.use_LSTM:
        lstm_str = 'lstm_'

    agent = algorithms.a3c.worker.Factory(
        params=params,
        master=algorithms.a3c.master.Stub(grpc.insecure_channel('localhost:50051')),
        log_dir='logs/boxing_a3c_%s%dthreads/worker_%d' % (lstm_str, 1, n_worker)
    )()

    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        message = receive(s)
        if message is None:
            break
        # info('message %s', str(message)[:64])
        verb, args = message
        if verb == 'act':
            state = json.loads(args[0], object_hook=_ndarray_decoder)
            send(s, 'act', agent.act(state))
        if verb == 'reward_and_reset':
            score = agent.reward_and_reset(args[0])
            if score is None:
                break
            send(s, 'reset', score)
        if verb == 'reward_and_act':
            state = json.loads(args[1], object_hook=_ndarray_decoder)
            action = agent.reward_and_act(args[0], state)
            if action is None:
                break
            send(s, 'act', action)

    info('%s: stop', address)


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=logging.INFO
    )

    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    s = socket.socket()
    s.bind(('localhost', 7000))
    s.listen(100)

    n_worker = 0

    while True:
        c, address = s.accept()
        pid = os.fork()
        if pid == 0:
            s.close()
            handle_connection(c, address, n_worker)
            c.close()
            break
        c.close()
        n_worker += 1

    s.close()


def send(socket, verb, *args):
    sends(socket, verb)
    sends(socket, json.dumps(args))

def sends(socket, string):
    assert type(string) == type('')
    socket.sendall(struct.pack('<I', len(string)))
    socket.sendall(string)

class ReceiveError(Exception):
    pass
        
def receive(socket):
    try:
        return receives(socket), json.loads(receives(socket))
    except ReceiveError:
        return None

def receives(socket):
    l1 = len(struct.pack('<I', 0))
    l2 = struct.unpack('<I', receiveb(socket, l1))[0]
    return receiveb(socket, l2)

def receiveb(socket, length):
    packets = []
    rest = length
    while rest > 0:
        packet = socket.recv(rest)
        if not packet:
            raise ReceiveError
        packets.append(packet)
        rest -= len(packet)
    data = ''.join(packets)
    assert len(data) == length
    return data


def _ndarray_decoder(dct):
    """Decoder from base64 to numpy.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return numpy.load(output)['obj']
    return dct


if __name__ == '__main__':
    main()
