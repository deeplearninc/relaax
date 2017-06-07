#!/usr/bin/env python
from __future__ import print_function
from builtins import object
import traceback

from relaax.client.rlx_client_config import options
from relaax.client.rlx_client import RlxClient, RlxClientException
from lab import LabEnv


class Training(object):

    def __init__(self):
        self.lab = LabEnv()
        self.max_episodes = options.get('environment/max_episodes', 1000)
        self.infinite_run = options.get('environment/infinite_run', False)
        self.agent = RlxClient(options.get('relaax_rlx_server/bind', 'localhost:7001'))

    def run(self):
        try:
            # connect to RLX server
            self.agent.connect()
            # give agent a moment to load and initialize
            self.agent.init(options.get('exploit', False))

            episode_cnt = 0
            while (episode_cnt < self.max_episodes) or self.infinite_run:
                try:
                    state = self.lab.reset()
                    reward, episode_reward, terminal = None, 0, False  # reward = 0 | None
                    action = self.agent.update(reward, state, terminal)
                    while not terminal:
                        reward, state, terminal = self.lab.act(action)
                        action = self.agent.update(reward, state, terminal)
                        episode_reward += reward
                    episode_cnt += 1
                    print('Episode:', episode_cnt, '| reward:', episode_reward)
                    self.agent.metrics.scalar('Game-Score', episode_reward)

                except RlxClientException as e:
                    print("agent connection lost: ", e)
                    print('reconnecting to another agent, '
                          'retrying to connect 10 times before giving up...')
                    try:
                        self.agent.connect(retry=10)
                        self.agent.init()
                        continue
                    except:
                        print("Can't reconnect, exiting")
                        raise Exception("Can't reconnect")

        except Exception as e:
            print("Something went wrong: ", e)
            traceback.print_exc()

        finally:
            # disconnect from the server
            self.agent.disconnect()

if __name__ == '__main__':
    Training().run()
