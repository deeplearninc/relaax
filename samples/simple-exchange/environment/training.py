from __future__ import print_function
from builtins import range

import random
import logging

from relaax.environment.config import options
from relaax.environment.training import TrainingBase

log = logging.getLogger(__name__)


class Training(TrainingBase):

    def __init__(self):
        super(Training, self).__init__()
        self.episode_length = options.get('environment/episode_length', 5)

    def actor(self, action):
        if random.random() >= 0.5:
            state = [1, 0]
        else:
            state = [0, 1]
        reward = (action[0] - state[0]) ** 2
        return reward, state

    def episode(self, number):
        # get first action from agent
        action = self.agent.update(reward=None, state=[1, 0])
        log.info('action: %s' % action)

        episode_reward = 0
        for episode_step in range(self.episode_length):
            reward, state = self.actor(action)
            episode_reward += reward
            # update agent with state and reward
            action = self.agent.update(reward=reward, state=state)
            log.info('action: %s' % action)
        return episode_reward


if __name__ == '__main__':
    Training().run()
