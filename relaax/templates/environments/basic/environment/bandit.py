import numpy as np


class Bandit(object):

    def __init__(self):
        # List out our bandits.
        # Default bandit 4 (index#3) is set to most often provide a positive reward.
        self.slots = [0.2, 0, -0.2, -5]

    def pull(self, action):
        result = np.random.randn(1)
        if result > self.slots[action]:
            return 1
        else:
            return -1
