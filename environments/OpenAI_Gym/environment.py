from __future__ import print_function

import logging
import os
import random
import time
import signal

from relaax.client import rlx_client

from . import game_process


def run(rlx_server_url, env, seed, limit):
    n_game = 0
    game = game_process.GameProcessFactory(env, limit).new_env(_seed(seed))

    def toggle_rendering():
        if game.display:
            game._close_display = True
        else:
            game.display = True
            game._close_display = False

    signal.signal(signal.SIGUSR1, lambda _1, _2: toggle_rendering())
    signal.siginterrupt(signal.SIGUSR1, False)

    while True:
        try:
            client = rlx_client.Client(rlx_server_url)
            try:
                action = client.init(game.state())
                while True:
                    reward, reset = game.act(action)
                    if reset:
                        episode_score = client.reset(reward)
                        n_game += 1
                        print('Score at game', n_game, '=', episode_score)
                        game.reset()
                        action = _send(client, None, game.state())
                    else:
                        action = _send(client, reward, game.state())
            finally:
                client.disconnect()
        except rlx_client.Failure as e:
            _warning('{} : {}'.format(rlx_server_url, e.message))
            delay = random.randint(1, 10)
            _info('waiting for %ds...', delay)
            time.sleep(delay)


def _send(client, reward, state):
    start = time.time()
    action = client.send(reward, state)
    client.metrics().scalar('client latency', time.time() - start)
    return action


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)
