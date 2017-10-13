from __future__ import print_function

import logging

from relaax.environment.training import TrainingBase

from lab import LabEnv

log = logging.getLogger(__name__)


class Training(TrainingBase):

    def __init__(self):
        super(Training, self).__init__()
        self.lab = LabEnv()

    def episode(self, number):
        state = self.lab.reset()
        reward, episode_reward, terminal = None, 0, False
        action = self.agent.update(reward, state, terminal)

        while not terminal:
            reward, state, terminal = self.lab.act(action)
            action = self.agent.update(reward, state, terminal)
            episode_reward += reward

        log.info('*********************************\n'
                 '*** Episode: %s * reward: %s \n'
                 '*********************************\n' %
                 (number, episode_reward))

        return episode_reward

if __name__ == '__main__':
    Training().run()
