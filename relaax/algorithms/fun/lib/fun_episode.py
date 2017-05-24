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

        # addition last fields
        self.last_zt_inp = None
        self.last_m_value = None
        self.last_goal = None

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
        if reward is not None:
            reward_i = 0
            self.push_experience(reward, reward_i)
        assert (state is None) == terminal

        if state is not None:   # except terminal
            self.states.append(state)   # also as first state
            self.last_zt_inp = self.session.op_get_zt(ph_state=[state])

            goal, self.last_m_value,\
                s_t, lstm_state = self.session.op_get_goal_value_st(
                    ph_perception=self.last_zt_inp,
                    ph_initial_lstm_state=self.session.op_get_lstm_state,
                    ph_step_size=[1])
            self.session.op_assign_lstm_state(ph_variable=lstm_state)

            self.goal_buffer.extend(goal)
            self.last_goal = self.goal_buffer.get_sum()
            self.st_buffer.extend(s_t)

        assert self.last_action is None
        assert self.last_value is None

        self.get_action_and_value()

    def end(self):
        experience = self.episode.end()
        if not self.exploit:
            self.apply_gradients(self.compute_gradients(experience), len(experience))

    def reset(self):
        self.episode = episode.Episode('state', 'action', 'reward', 'value',
                                       'zt_inp', 'goal', 'reward_i', 'm_value')
    # Helper methods

    def push_experience(self, reward, reward_i):
        assert len(self.states) != 0
        assert self.last_action is not None
        assert self.last_value is not None

        self.episode.step(
            reward=reward,
            reward_i=reward_i,
            state=self.states,
            action=self.last_action,
            value=self.last_value,
            m_value=self.last_m_value,
            goal=self.last_goal
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
        action, value = self.session.op_get_action_and_value(state=self.states)
        probabilities, = action
        value, = value
        return utils.choose_action(probabilities), value

    def compute_gradients(self, experience):
        r = ri = 0.0
        if self.last_value is not None:
            r = self.last_value
            # need last z_t to compute ri
            ri = z_t = 0.0

        # shift to cfg.c = 10  (could be at the end method)

        reward = experience['reward']
        m_reward = experience['reward_i']
        discounted_reward = np.zeros_like(reward, dtype=np.float32)
        m_discounted_reward = np.zeros_like(m_reward, dtype=np.float32)

        # compute and accumulate gradients
        for t in reversed(range(len(reward))):
            r = reward[t] + cfg.worker_gamma * r
            ri = m_reward[t] + cfg.manager_gamma * ri
            discounted_reward[t] = r
            m_discounted_reward[t] = ri

        step_sz = len(experience['action'])
        st_diff = self.st_buffer.get_diff(part=step_sz)

        manager_gradients = self.session.op_compute_gradients(
                    ph_perception=experience['zt_inp'],
                    ph_stc_diff_st=st_diff,
                    ph_discounted_reward=m_discounted_reward,
                    ph_initial_lstm_state=self.manager_start_lstm_state,
                    ph_step_size=step_sz)

        worker_gradients = self.session.op_compute_gradients(
                    ph_state=experience['state'],
                    ph_goal=experience['goal'],
                    ph_action=experience['action'],
                    ph_value=experience['value'],
                    ph_discounted_reward=discounted_reward + m_discounted_reward,
                    ph_initial_lstm_state=self.worker_start_lstm_state,
                    ph_step_size=step_sz)
        return manager_gradients, worker_gradients

    def apply_gradients(self, gradients, experience_size):
        self.ps.session.op_apply_gradients(
            gradients=gradients[0],  # manager
            increment=experience_size
        )
        self.ps.session.op_apply_gradients(
            gradients=gradients[1],  # worker
            increment=experience_size
        )
