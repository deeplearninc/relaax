import numpy as np
import tensorflow as tf

from pg_config import config
from pg_network import PolicyNN


# PGAgent implements training regime for Policy Gradient algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class PGAgent(object):

    def __init__(self, parameter_server):
        self.ps = parameter_server

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        self.exploit = exploit
        # count global steps between all agents
        self.global_t = 0
        # count of episodes run by agent
        self.local_t = 0
        # reset variables used
        # to run single episode
        self.reset_episode()
        # Build TF graph
        self.nn = PolicyNN()
        # Initialize TF
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):

        # beginning of episode
        if self.episode_t == 0:
            self.begin_episode()

        # every episode step
        # will increase episode_t
        # by one if state is not None
        action = self.episode_step(reward, state)

        # end_episode will set episode_t to 0
        if (self.episode_t == config.batch_size) or terminal:
            self.end_episode()

        return action

    # environment is asking to reset agent
    def reset(self):
        self.reset_episode()
        return True

# Episode states

    # load shared parameters from PS
    def begin_episode(self):
        self.load_shared_parameters()

    # every episode step calculate action
    # by running policy and accumulate experience
    def episode_step(self, reward, state):
        # if state is None then skipping this episode
        # there is no action for None state
        if state is None:
            return None

        if state.ndim > 1:
            state = state.flatten()
        action = self.action_from_policy(state)
        # accumulate experience
        self.accumulate(state, reward, action)

        return action

    # train policy on accumulated experience
    # and update shared NN parameters
    def end_episode(self):
        if (self.episode_t > 1) and (not self.exploit):
            partial_gradients = self.train_policy()
            self.update_shared_parameters(partial_gradients)
        self.reset_episode()
        # increase number of completed episodes
        self.local_t = 0

# Helper methods

    # train policy with accumulated states, rewards and actions
    def train_policy(self):
        def discounted_reward(self):
            # take 1D float array of rewards and compute discounted reward
            rewards = np.vstack(self.rewards)
            discounted_reward = np.zeros_like(rewards)
            running_add = 0
            for t in reversed(xrange(0, rewards.size)):
                running_add = running_add * config.GAMMA + rewards[t]
                discounted_reward[t] = running_add
            # size the rewards to be unit normal
            # it helps control the gradient estimator variance
            discounted_reward = discounted_reward.astype(np.float64)
            discounted_reward -= np.mean(discounted_reward)
            discounted_reward /= np.std(discounted_reward) + 1e-20
            return discounted_reward

        return self.sess.run(self.nn.partial_gradients, feed_dict={
                             self.nn.state: self.states,
                             self.nn.action: self.actions,
                             self.nn.discounted_reward: discounted_reward()})

    # update PS with leaned policy
    def update_shared_parameters(self, partial_gradients):
        self.ps.run(
            "apply_gradients", feed_dict={"gradients": partial_gradients})

    #
    def reset_episode(self):
        self.episode_reward, self.episode_t = 0, 0
        self.rewards, self.states, self.actions = [], [], []

    # reload policy weights from PS
    def load_shared_parameters(self):
        weights = self.ps.run("weights")
        self.sess.run(
            self.nn.assign_weights,
            feed_dict={self.nn.shared_wights: weights})

    # run agent's policy and get action
    def action_from_policy(self, state):
        def choose_action(probabilities):
            values = np.cumsum(probabilities)
            r = np.random.rand()*values[-1]
            return np.searchsorted(values, r)

        if state:
            action_probabilities = self.sess.run(
                self.nn.policy, feed_dict={self.nn.state: [state]})
            return choose_action(action_probabilities)
        return None

    # accumulate experience:
    # state, reward, and actions for policy training
    def accumulate(self, state, reward, action):
        self.state.append(state)

        # one-hot vector to store taken action
        action_vec = np.zeros([config.action_size])
        action_vec[action] = 1
        self.actions.append(action_vec)

        if reward is None:
            reward = 0
        self.episode_reward += reward
        self.rewards.append(reward)
