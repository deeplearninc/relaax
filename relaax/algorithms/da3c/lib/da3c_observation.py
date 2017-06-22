from builtins import object
import numpy as np

from .. import da3c_config


class DA3CObservation(object):
    def __init__(self):
        self.queue = None

    def add_state(self, state):
        if state is None:
            self.queue = None
            return

        if da3c_config.config.input.history == 1:
            self.queue = state
            return

        state = np.asarray(state)
        axis = len(state.shape)  # extra dimension for observation
        observation = np.reshape(state, state.shape + (1,))
        if self.queue is None:
            self.queue = np.repeat(observation,
                    da3c_config.config.input.history, axis=axis)
        else:
            # remove oldest observation from the begining of the observation queue
            self.queue = np.delete(self.queue, 0, axis=axis)

            # append latest observation to the end of the observation queue
            self.queue = np.append(self.queue, observation, axis=axis)
