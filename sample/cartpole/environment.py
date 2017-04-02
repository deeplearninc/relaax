from __future__ import print_function

import logging
import os
import random
import signal
from time import time, sleep

from relaax.client import rlx_client

import game_process


def run():
    rlx_server_url = 'localhost:7001'
    n_game = 0
    game = game_process.GameProcessFactory().new_env()

    def toggle_rendering():
        if game.display:
            game._close_display = True
        else:
            game.display = True
            game._close_display = False

    signal.signal(signal.SIGUSR1, lambda _1, _2: toggle_rendering())
    signal.siginterrupt(signal.SIGUSR1, False)

    # create client instance
    client = rlx_client.RlxClient(rlx_server_url)
    try:
        # connect to the server
        client.connect()

        # give agent a moment to load and initialize
        res = client.init()
        print("on init: ", res)

        while True:
            game.reset()
            state = game.state()
            reward = 0  #reward = None
            terminal = False
            while not terminal:
                    # if it is terminal state, set terminal to True
                    action = client.update(reward, state, terminal)

                    reward, terminal = game.act(action)
                    state = game.state()
                    print("action:", action)

        # reset agent
        res = client.reset()
        print("on reset:", res)

    except Exception as e:
        print("Something went wrong: ", e)
        raise

    finally:
        # disconnect from the server
        if client:
            client.disconnect()


def _send(client, reward, state):
    action = client.send(reward, state)
    return action


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)
