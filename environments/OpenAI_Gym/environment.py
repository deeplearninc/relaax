from __future__ import print_function

import logging
import os
import random
import socket
import time
import signal

from . import game_process
from relaax import client


def run(rlx_server, env, seed):
    server_address = rlx_server
    game = game_process.GameProcessFactory(env).new_env(_seed(seed))

    def toggle_rendering():
        if game.display:
            game._close_display = True
        else:
            game.display = True
            game._close_display = False

    signal.signal(signal.SIGUSR1, lambda _1, _2: toggle_rendering())
    signal.siginterrupt(signal.SIGUSR1, False)

    while True:
        s = socket.socket()
        try:
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _connectf(s, _parse_address(server_address))
                c = client.SocketClient(s)
                action = c.init(game.state())
                while True:
                    reward, reset = game.act(action)
                    if reset:
                        episode_score = c.reset(reward)
                        n_game += 1
                        print('Score at game', n_game, '=', episode_score)
                        game.reset()
                        assert reward is None
                    action = c.send(reward, game.state())
            finally:
                s.close()
        except client.Failure as e:
            _warning('{} : {}'.format(server_address, e.message))
            delay = random.randint(1, 10)
            _info('waiting for %ds...', delay)
            time.sleep(delay)


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value


def _parse_address(address):
    host, port = address.split(':')
    return host, int(port)


def _connectf(s, server_address):
    try:
        s.connect(server_address)
    except socket.error as e:
        raise client.Failure("socket error({}): {}".format(e.errno, e.strerror))


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)
