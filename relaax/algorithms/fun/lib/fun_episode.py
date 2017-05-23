from __future__ import absolute_import
from builtins import range
from builtins import object
import numpy as np

from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from ..fun_config import config as cfg
from ..fun_model import LocalWorkerNetwork, LocalManagerNetwork
from .utils import RingBuffer2D


class FuNEpisode(object):
    def __init__(self, parameter_server, exploit):
        self.exploit = exploit
        self.ps = parameter_server
        models = [LocalManagerNetwork, LocalWorkerNetwork]
        self.session = session.Session(models)  # needs to support multiply models
        self.reset()

        self.goal_buffer = RingBuffer2D(element_size=cfg.d,
                                        buffer_size=cfg.c * 2)
        self.st_buffer = RingBuffer2D(element_size=cfg.d,
                                      buffer_size=cfg.c * 2)
        self.states = []
        self.last_action = None
        self.last_value = None
        self.first = cfg.c  # =batch_size

        self.worker_start_lstm_state = None
        self.manager_start_lstm_state = None
        self.cur_c = 0

        # while not in experience pushing
        self.zt_inp = []

    @property
    def experience(self):
        return self.episode.experience

    def begin(self):
        self.load_shared_parameters()
        # my beg ==before main loop==
        if self.first == 0:
            self.update_buffers()
        self.store_lstm_states()
        # my end ==before main loop==
        self.get_action_and_value()
        self.episode.begin()

    def step(self, reward, state, terminal):
        z_t = self.session.op_get_zt(ph_state=[state])
        self.zt_inp.append(z_t)

        if reward is not None:
            self.push_experience(reward)
        assert (state is None) == terminal
        self.observation.add_state(state)

        assert self.last_action is None
        assert self.last_value is None

        self.get_action_and_value()

    def end(self):
        experience = self.episode.end()
        if not self.exploit:
            self.apply_gradients(self.compute_gradients(experience), len(experience))

    def reset(self):
        self.episode = episode.Episode('state', 'action', 'reward', 'value',
                                       'goal', 'm_value', 'state_t', 'reward_i', 'zt_inp')
    # Helper methods

    def push_experience(self, reward):
        assert len(self.states) != 0
        assert self.last_action is not None
        assert self.last_value is not None

        self.episode.step(
            reward=reward,
            state=self.states,
            action=self.last_action,
            value=self.last_value
        )
        self.last_action = None
        self.last_value = None

    def update_buffers(self):
        zt_batch = self.session.op_get_zt(ph_state=self.states)

        goals_batch, st_batch = self.session.op_get_goal_st(
            ph_perception=zt_batch,
            ph_initial_lstm_state=self.sg_network.lstm_state_out,
            ph_step_size=cfg.c)

        # second half is used for intrinsic reward calculation
        self.goal_buffer.replace_second_half(goals_batch)
        self.st_buffer.replace_second_half(st_batch)

    def store_lstm_states(self):
        # need to split models within session
        self.worker_start_lstm_state = self.session.op_get_lstm_state
        self.manager_start_lstm_state = self.session.op_get_lstm_state

    def get_action_and_value(self):
        if len(self.states) == 0:
            self.last_action = None
            self.last_value = None
        else:
            self.last_action, self.last_value = self.get_action_and_value_from_network()
            assert self.last_action is not None
            assert self.last_value is not None

    def keep_action_and_value(self, action, value):
        assert self.last_action is None
        assert self.last_value is None

        self.last_action = action
        self.last_value = value

    def load_shared_parameters(self):
        self.session.op_assign_weights(weights=self.ps.session.op_get_weights())
        # assume that we sync both manager & worker

    def get_action_and_value_from_network(self):
        action, value = self.session.op_get_action_and_value(state=[self.observation.queue])
        probabilities, = action
        value, = value
        return utils.choose_action(probabilities), value

    def compute_gradients(self, experience):
        r = 0.0
        if self.last_value is not None:
            r = self.last_value

        reward = experience['reward']
        discounted_reward = np.zeros_like(reward, dtype=np.float32)

        # compute and accumulate gradients
        for t in reversed(range(len(reward))):
            r = reward[t] + cfg.rewards_gamma * r
            discounted_reward[t] = r

        return self.session.op_compute_gradients(
            state=experience['state'],
            action=experience['action'],
            value=experience['value'],
            discounted_reward=discounted_reward
        )

    def apply_gradients(self, gradients, experience_size):
        self.ps.session.op_apply_gradients(
            gradients=gradients,
            increment=experience_size
        )
