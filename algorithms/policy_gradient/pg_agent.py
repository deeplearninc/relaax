import tensorflow as tf

from pg_config import config
from pg_network import PolicyNN

from relaax.common.algorithms.accums import accumulate
from relaax.common.algorithms.acts import action_from_policy
from relaax.common.algorithms.sync import load_shared_parameters, update_shared_parameters
from relaax.common.algorithms.train import train_policy


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
        load_shared_parameters(self)

    # every episode step calculate action
    # by running policy and accumulate experience
    def episode_step(self, reward, state):
        # if state is None then skipping this episode
        # there is no action for None state
        if state is None:   # terminal is reached
            self.rewards.append(reward)
            self.episode_t += 1
            return None

        if state.ndim > 1:
            state = state.flatten()
        probs = action_from_policy(self, state)

        # accumulate experience & retrieve action as simple number
        action = accumulate(self, state, reward, probs)

        return action

    # train policy on accumulated experience
    # and update shared NN parameters
    def end_episode(self):
        if (self.episode_t > 1) and (not self.exploit):
            partial_gradients = train_policy(self)
            update_shared_parameters(self, partial_gradients)
        self.reset_episode()

    # reset training auxiliary counters and accumulators
    # (also needs to create auxiliary members -> don't move)
    def reset_episode(self):
        self.episode_reward, self.episode_t = 0, 0
        self.rewards, self.states, self.actions = [], [], []
