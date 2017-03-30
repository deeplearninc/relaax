import numpy as np


class Experience(object):

    def __init__(self, action_size):
        self.action_size = action_size
        self.reset()

    # accumulate experience:
    # state, reward, and actions for policy training
    def accumulate(self, state, reward, action):
        self.state.append(state)

        # one-hot vector to store taken action
        action_vec = np.zeros([self.action_size])
        action_vec[action] = 1
        self.actions.append(action_vec)

        if reward is None:
            reward = 0
        self.episode_reward += reward
        self.rewards.append(reward)

    def reset(self):
        self.states = []
        self.rewards = []
        self.actions = []
