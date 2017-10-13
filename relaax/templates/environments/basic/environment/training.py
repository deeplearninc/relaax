from __future__ import print_function
from builtins import range

import logging

from relaax.environment.config import options
from relaax.environment.training import TrainingBase

from bandit import Bandit

log = logging.getLogger(__name__)


class Training(TrainingBase):

    def __init__(self):
        super(Training, self).__init__()
        self.steps = options.get('environment/steps', 1000)
        self.bandit = Bandit()

    def episode(self, number):
        # get first action from agent
        action = self.agent.update(reward=None, state=[])
        # update agent with state and reward
        for step in range(self.steps):
            reward = self.bandit.pull(action)
            action = self.agent.update(reward=reward, state=[])
            log.info('step: %s, action: %s' % (step, action))


if __name__ == '__main__':
    Training().run()
