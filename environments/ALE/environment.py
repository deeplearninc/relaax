from __future__ import print_function

import logging
import os
import random
import time

from relaax.client import rlx_client_config
from relaax.client import rlx_client

import game_process


def run():
    rlx_server_url = rlx_client_config.options.get('environment/rlx_server')
    rom = rlx_client_config.options.get('environment/rom')
    display = False
    seed = None

    n_game = 0
    game = game_process.GameProcessFactory(rom, display).new_env(_seed(seed))

    agent = rlx_client.RlxClient(rlx_server_url)
    assert agent is not None
    try:
        agent.connect()
        agent.init()
        episode_reward = 0
        reward = None
        while True:
            try:
                action = agent.update(reward=reward, state=game.state(), terminal=False)
                print('ACTION', repr(action))
                reward, reset = game.act(action['data'])
                if reward is not None:
                    episode_reward += reward
                if reset:
                    agent.update(reward=reward, state=None, terminal=True)
                    reward = None
                    n_game += 1
                    print('Score at game', n_game, '=', episode_reward)
                    game.reset()
                    episode_reward = 0
            except rlx_client.RlxClientException as e:
                print("agent error: ", e)
                game.reset()
                agent.connect(retry=10)
                agent.init()
                episode_reward = 0
                reward = None

        agent.reset()

    finally:
        agent.disconnect()


def _seed(value):
    if value is not None:
        return value
    return random.randrange(1000000)
