class Experience(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def length(self):
        assert len(self.states) == len(self.actions)
        assert len(self.states) == len(self.rewards)
        return len(self.states)

    # accumulate experience:
    # state, reward, and actions for policy training
    def accumulate(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
