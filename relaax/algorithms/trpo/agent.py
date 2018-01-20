from __future__ import absolute_import

from builtins import object
from collections import defaultdict
import logging
import numpy as np
import time

from relaax.server.common import session
from relaax.common import profiling
from relaax.common.algorithms.lib import observation

from . import trpo_config
from . import trpo_model
from .lib import network


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        model = trpo_model.AgentModel()
        self.session = session.Session(model)

        self._episode_timestep = 0   # timestep for current episode (round)
        self._episode_reward = 0     # score accumulator for current episode (round)
        self._stop_training = False  # stop training flag to prevent the training further

        self.data = defaultdict(list)
        self.observation = observation.Observation(trpo_config.config.input.history)

        self.policy = network.make_policy_wrapper(self.session, self.ps.metrics)

        # counter for global updates at parameter server
        self._n_iter = self.ps.session.call_wait_for_iteration()
        self.session.op_set_weights(weights=self.ps.session.policy.op_get_weights())

        if trpo_config.config.use_filter:
            self.obs_filter = network.make_filter(trpo_config.config)
            state = self.ps.session.call_get_filter_state()
            self.obs_filter.rs.set(*state)

        self.server_latency_accumulator = 0     # accumulator for averaging server latency
        self.collecting_time = time.time()      # timer for collecting experience

        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    @profiler.wrap
    def update(self, reward, state, terminal):
        self.check_state_shape(state)
        # replace empty state with constant one
        if list(np.asarray(state).shape) == [0]:
            state = [0]
        if reward is not None:
            if self.reward(reward):
                return None
        if terminal:
            self.reset()
            return None
        assert state is not None
        return self.act(np.asarray(state))

    @staticmethod
    def check_state_shape(state):
        if state is None:
            return
        expected_shape = list(trpo_config.options.algorithm.input.shape)
        actual_shape = list(np.asarray(state).shape)
        if actual_shape != expected_shape:
            logger.warning('State shape %s does not match to expected one %s.', repr(actual_shape),
                           repr(expected_shape))

    def act(self, state):
        start = time.time()

        if trpo_config.config.use_filter:
            state = self.obs_filter(state)

        self.observation.add_state(state)
        self.data["observation"].append(self.observation.queue)

        action, agentinfo = self.policy.act([self.observation.queue])
        self.data["action"].append(action)

        for (k, v) in agentinfo.items():
            self.data[k].append(v)

        self.server_latency_accumulator += time.time() - start
        return action

    def reward_and_act(self, reward, state):
        if not self.reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if self.reward(reward):
            return None
        return self.reset()

    def reset(self):
        latency = self.server_latency_accumulator / self._episode_timestep
        self.server_latency_accumulator = 0
        self.ps.metrics.scalar('server_latency', latency)

        self.observation.add_state(None)
        self._send_experience(terminated=(self._episode_timestep <
                                          trpo_config.config.PG_OPTIONS.timestep_limit))
        return True

    def reward(self, reward):
        self._episode_reward += reward

        # reward = self.reward_filter(reward)
        self.data["reward"].append(reward)

        self._episode_timestep += 1

        return self._stop_training

    def _send_experience(self, terminated=False):
        self.data["terminated"] = terminated
        self.data["filter_diff"] = (0, np.zeros(1), np.zeros(1))
        if trpo_config.config.use_filter:
            mean, std = self.obs_filter.rs.get_diff()
            self.data["filter_diff"] = (self._episode_timestep, mean, std)
        self.ps.session.call_send_experience(self._n_iter, dict(self.data), self._episode_timestep)

        self.data.clear()
        self._episode_timestep = 0
        self._episode_reward = 0

        old_n_iter = self._n_iter
        self._n_iter = self.ps.session.call_wait_for_iteration()
        if self._n_iter == -1:
            self._n_iter = old_n_iter
            return

        if self._n_iter > trpo_config.config.PG_OPTIONS.n_iter:
            self._stop_training = True
            return

        if old_n_iter < self._n_iter:
            print('Collecting time for {} iteration: {}'.format(old_n_iter+1, time.time() -
                  self.collecting_time))
            self.session.op_set_weights(weights=self.ps.session.policy.op_get_weights())
            self.collecting_time = time.time()

        if trpo_config.config.use_filter:
            state = self.ps.session.call_get_filter_state()
            self.obs_filter.rs.set(*state)
