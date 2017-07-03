from __future__ import print_function

import logging

from relaax.environment.config import options
from relaax.environment.training import TrainingBase

from gym_env import GymEnv

log = logging.getLogger(__name__)


class Training(TrainingBase):

    def __init__(self):
        super(Training, self).__init__()
        self.gym = GymEnv(env=options.get('environment/name', 'CartPole-v0'))

    def episode(self, number):
        state = self.gym.reset()
        reward, episode_reward, terminal = None, 0, False
        action = self.agent.update(reward, state, terminal)
        while not terminal:
            reward, state, terminal = self.gym.act(action)
            action = self.agent.update(reward, state, terminal)
            episode_reward += reward
        log.info('Episode %d reward %d' % (number, episode_reward))
        return episode_reward

if __name__ == '__main__':
    Training().run()
