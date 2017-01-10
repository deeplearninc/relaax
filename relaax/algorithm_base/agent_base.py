

class AgentBase(object):
    def act(self, state):
        raise NotImplementedError

    def reward_and_reset(self, reward):
        raise NotImplementedError

    def reward_and_act(self, reward, state):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError
