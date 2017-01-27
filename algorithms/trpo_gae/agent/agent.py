import tensorflow as tf
import time
import itertools
from collections import defaultdict

import relaax.algorithm_base.agent_base
import relaax.common.protocol.socket_protocol

from . import network


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    # TODO: implement all abstract methods
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server

        self.global_t = 0           # counter for global steps between all agents
        self.local_t = 0            # steps count for current agent
        self.episode_reward = 0     # score accumulator for current episode (round)

        self.policy_net, self.value_net = network.make(config)
        self.obs_filter, self.reward_filter = network.make_filters(config)

        self.paths = []
        self.data = defaultdict(list)
        self.seed_iter = itertools.count()

        initialize_all_variables = tf.variables_initializer(tf.global_variables())
        self._session = tf.Session()

        self.policy, self.baseline = network.make_head(config, self.policy_net, self.value_net, self._session)

        self._session.run(initialize_all_variables)

    def act(self, state):
        start = time.time()

        obs = self.obs_filter(state)
        self.data["observation"].append(obs)

        action, agentinfo = self.policy.act(obs)

        self.data["action"].append(action)
        for (k, v) in agentinfo.iteritems():
            self.data[k].append(v)

        self.metrics().scalar('server latency', time.time() - start)

        return action

    # do_rollouts_serial(env, agent, timestep_limit, n_timesteps, seed_iter)
    # compute_advantage(vf, paths, gamma, lam):

    def reward_and_act(self, reward, state):
        if self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if not self._reward(reward):
            return None

    def _reward(self, reward):
        self.episode_reward += reward

        self.local_t += 1
        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step

    def metrics(self):
        return self._parameter_server.metrics()