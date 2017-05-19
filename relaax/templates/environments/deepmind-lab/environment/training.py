from __future__ import print_function

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
        # try:
        #     # connect to RLX server
        #     self.agent.connect()
        #     # give agent a moment to load and initialize
        #     self.agent.init()

        #     episode_cnt = 0
        #     while (episode_cnt < self.max_episodes) or self.infinite_run:
        #         try:
        #             self.gym.reset()
        #             state = self.gym.state()
        #             reward, episode_reward = None, 0  # reward = 0 | None
        #             terminal = False
        #             action = self.agent.update(reward, state, terminal)
        #             while not terminal:
        #                 reward, terminal = self.gym.act(action['data'])
        #                 episode_reward += reward
        #                 state = None if terminal else self.game.state()
        #                 action = self.agent.update(reward, state, terminal)
        #             episode_cnt += 1
        #             print('Game:', episode_cnt, '| Episode reward:', episode_reward)
        #         except RlxClientException as e:
        #             print("agent connection lost: ", e)
        #             print('reconnecting to another agent, '
        #                   'retrying to connect 10 times before giving up...')
        #             try:
        #                 self.agent.connect(retry=10)
        #                 self.agent.init()
        #                 continue
        #             except:
        #                 print("Can't reconnect, exiting")
        #                 raise Exception("Can't reconnect")

        # except Exception as e:
        #     print("Something went wrong: ", e)

        # finally:
        #     # disconnect from the server
        #     self.agent.disconnect()
        pass

if __name__ == '__main__':
    Training().run()
