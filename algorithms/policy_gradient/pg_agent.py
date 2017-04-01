from relaax.server.common.session import Session

from lib.experience import Experience
from lib.utils import discounted_reward, choose_action

from pg_config import config
from pg_model import PolicyModel


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
        # experience accumulated through episode
        self.experience = Experience(config.action_size)
        # reset variables used
        # to run single episode
        self.reset_episode()
        # Build TF graph
        self.model = PolicyModel()
        # Initialize TF
        self.sess = Session(self.model)

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

        action = self.action_from_policy(state)
        self.experience.accumulate(state, reward, action)
        self.episode_t += 1

        return action

    # train policy on accumulated experience
    # and update shared NN parameters
    def end_episode(self):
        if (self.episode_t > 1) and (not self.exploit):
            partial_gradients = self.train_policy()
            self.update_shared_parameters(partial_gradients)
        self.reset_episode()

    # reset training auxiliary counters and accumulators
    # (also needs to create auxiliary members -> don't move)
    def reset_episode(self):
        self.experience.reset()
        self.episode_reward, self.episode_t = 0, 0

# Helper methods

    # reload policy weights from PS
    def load_shared_parameters(self):
        weights, = self.ps.run([self.ps.graph.weights])
        self.sess.run(
            [self.sess.graph.assign_weights],
            feed_dict={self.sess.graph.shared_weights: weights}
        )

    # run policy and get action
    def action_from_policy(self, state):
        if state is None:
            return None
        action_probabilities, = self.sess.run(
            [self.sess.graph.policy],
            feed_dict={self.sess.graph.state: [state]}
        )
        return choose_action(action_probabilities)

    # train policy with accumulated states, rewards and actions
    def train_policy(self):
        partial_gradients, = self.sess.run(
            [self.sess.graph.partial_gradients],
            feed_dict={
                self.sess.graph.state: self.experience.states,
                self.sess.graph.action: self.experience.actions,
                self.sess.graph.discounted_reward: discounted_reward(
                    self.experience.rewards,
                    config.GAMMA
                )
            }
        )
        return partial_gradients

    # update PS with learned policy
    def update_shared_parameters(self, partial_gradients):
        self.ps.run(
            [self.ps.graph.apply_gradients],
            feed_dict={self.ps.graph.gradients: partial_gradients}
        )
