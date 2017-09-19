from builtins import object
import numpy as np


class Observation(object):
    def __init__(self, stacked_num=1):
        self.stacked_num = stacked_num
        self.queue = None

    def add_state(self, state):
        if state is None:
            self.queue = None
            return

        state = np.asarray(state)
        axis = len(state.shape)  # extra dimension for observation
        observation = np.reshape(state, state.shape + (1,))
        if self.queue is None:
            self.queue = np.repeat(observation, self.stacked_num, axis=axis)
        else:
            # remove oldest observation from the beginning of the observation queue
            self.queue = np.delete(self.queue, 0, axis=axis)

            # append latest observation to the end of the observation queue
            self.queue = np.append(self.queue, observation, axis=axis)

    def get_state(self):
        return self.queue

    def is_none(self):
        return self.queue is None
