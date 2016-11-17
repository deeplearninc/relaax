from __future__ import print_function

import argparse
import base64
import io
import json
import logging
import numpy
import os
import random
import socket
import struct

import game_process
from .. import server_api


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument("--game", type=str, default="boxing", help="Name of the Atari game ROM")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator")
    return parser.parse_args()


def info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def main():
    args = parse_args()

    #logging.getLogger('requests').setLevel(logging.WARNING)
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    game = game_process.GameProcessFactory(args.game).new_env(_seed(args.seed))
    game_played = 0

    s = socket.socket()
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((args.host, args.port))

    send(s, 'act', dump_state(game))
    while True:
        message = receive(s)
        if message is None:
            break
        # info('message %s', str(message)[:64])
        verb, args = message
        if verb == 'act':
            reward, reset = game.act(args[0])
            if reset:
                send(s, 'reward_and_reset', reward)
            else:
                send(s, 'reward_and_act', reward, dump_state(game))
        if verb == 'reset':
            game_played += 1
            print("Score at game", game_played, "=", args[0])
            game.reset()
            send(s, 'act', dump_state(game))
    s.close()


def dump_state(game):
    return json.dumps(game.state(), cls=_NDArrayEncoder)


class _NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            output = io.BytesIO()
            numpy.savez_compressed(output, obj=obj)
            return {'b64npz': base64.b64encode(output.getvalue())}
        return json.JSONEncoder.default(self, obj)


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value


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


if __name__ == "__main__":
    main()
