from __future__ import print_function
from builtins import range

import logging
import numpy as np

from relaax.environment.config import options
from relaax.environment.training import TrainingBase


logger = logging.getLogger(__name__)


class Training(TrainingBase):
    def __init__(self):
        super(Training, self).__init__()
        self.steps = options.get('environment/steps', 1000)
        self.history_len = 2
        self.ring_buffer = RingBuffer(self.history_len)
        for _ in range(self.ring_buffer.size - 1):
            self.ring_buffer.append(0)

    def episode(self, number):
        state = self.state()
        action = self.agent.update(reward=None, state=state)
        for step in range(self.steps):
            reward = self.reward(action)
            print('state %s, action %s, reward %f' % (state, action, reward))
            state = self.state()
            action = self.agent.update(reward=reward, state=state)

    def state(self):
        state = np.random.random()
        self.ring_buffer.append(state)
        return [state]

    def reward(self, action):
        reward = self.history_len
        for i in range(self.history_len):
            reward -= abs(action[i] - self.ring_buffer[i])
        return reward


class RingBuffer(object):
    def __init__(self, size):
        self.size = size
        self._buffer = [None] * size
        self._index = 0

    def append(self, v):
        self._buffer[self._index] = v
        self._index = (self._index + 1) % self.size

    def __getitem__(self, i):
        return self._buffer[(self._index + i) % self.size]


if __name__ == '__main__':
    Training().run()
