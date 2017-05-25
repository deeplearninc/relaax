from __future__ import print_function

import logging
import os
import random
import signal
from time import time, sleep

from relaax.client import rlx_client_config
from relaax.client import rlx_client

from . import game_process


def run():
    rlx_server_url = rlx_client_config.options.get('environment/rlx_server')
    env = rlx_client_config.options.get('environment/env')
    seed = None
    limit = None
    rnd = 7
    frame_skip = None

    n_game = 0
    game = game_process.GameProcessFactory(env, rnd).new_env(_seed(seed), limit, frame_skip)

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
    if value is None:
        return random.randrange(1000000)
    return value
