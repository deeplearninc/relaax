from __future__ import print_function

import numpy as np
import random
import tensorflow as tf
import time

import relaax.algorithm_base.agent_base
import relaax.common.protocol.socket_protocol


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server

        self.global_t = 0           # counter for global steps between all agents
        self.local_t = 0            # step counter for current agent instance
        self.episode_reward = 0     # score accumulator for current episode (game)

        self.states = []            # auxiliary states accumulator through batch_size = 0..N
        self.actions = []           # auxiliary actions accumulator through batch_size = 0..N
        self.rewards = []           # auxiliary rewards accumulator through batch_size = 0..N

        self.episode_t = 0          # episode counter through batch_size = 0..M

        initialize_all_variables = tf.variables_initializer(tf.global_variables())

        self._session = tf.Session()

        self._session.run(initialize_all_variables)