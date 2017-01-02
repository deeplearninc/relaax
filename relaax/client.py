from __future__ import print_function


class Client(object):
    def act(self, state):
        raise NotImplementedError

    def reward_and_reset(self, reward):
        raise NotImplementedError

    def reward_and_act(self, reward, state):
        raise NotImplementedError
