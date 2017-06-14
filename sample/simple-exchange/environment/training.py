from __future__ import print_function
from builtins import range
from builtins import object

import random
import traceback

from relaax.environment.config import options
from relaax.environment.agent_proxy import AgentProxy, AgentProxyException


class Training(object):

    def __init__(self):
        self.agent = AgentProxy(options.get('relaax_rlx_server/bind', 'localhost:7001'))
        self.episodes = 10
        self.episode_length = 5

    def actor(self, action):
        if random.random() >= 0.5:
            state = [1, 0]
        else:
            state = [0, 1]
        reward = (action[0] - state[0]) ** 2
        return reward, state

    def run_episode(self, action):
        episode_reward = 0
        for episode_step in range(self.episode_length):
            reward, state = self.actor(action)
            episode_reward += reward
            # update agent with state and reward
            action = self.agent.update(reward=reward, state=state)
            print('action:', action)
        return episode_reward

    def run(self):
        try:
            # connect to the server
            self.agent.connect()
            # give agent a moment to load and initialize
            self.agent.init(options.get('exploit', False))
            # get first action from agent
            action = self.agent.update(reward=None, state=[1, 0])
            print('action:', action)

            # run training
            for step in range(self.episodes):
                try:
                    episode_reward = self.run_episode(action)
                    # send game score to accumulate running metrics
                    self.agent.metrics.scalar('game_score', episode_reward)

                except AgentProxyException as e:
                    print('Agent connection lost: ', e)
                    print('Reconnecting to another agent, '
                          'retrying to connect 10 times before giving up...')
                    try:
                        self.agent.connect(retry=10)
                        self.agent.init()
                        continue
                    except:
                        raise Exception('Can\'t reconnect, exiting...')

        except Exception as e:
            print("Something went wrong: ", e)
            traceback.print_exc()

        finally:
            # disconnect from the server
            self.agent.disconnect()


if __name__ == '__main__':
    Training().run()
