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
        self.session = session.Session(
            lm=LocalManagerNetwork(),
            lw=LocalWorkerNetwork()
        )
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
            if self.cur_c < cfg.c:
                self.cur_c += 1  # can replace

            if not terminal:
                cosine = self.get_cosine(state)
                reward_i = sum(cosine.diagonal()) / self.cur_c
            else:
                reward_i = 0

            self.push_experience(reward, reward_i)

        assert (state is None) == terminal

        if state is not None:   # except terminal
            self.states.append(state)   # also as first state
            self.manager_step(state)    # add manager's lasts & extend buffers

        assert self.last_action is None
        assert self.last_value is None

        self.get_action_and_value()  # similar to my local_nn.run_policy_and_value & choose

    def end(self):
        experience = self.episode.end()
        if len(experience['states']) > self.cfg.c:
            experience = dict(experience.items()[self.cfg.c:])
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

        goals_batch, st_batch, _ = self.session.op_get_goal_st(
            ph_perception=zt_batch,
            ph_initial_lstm_state=self.session.op_get_lstm_state,
            ph_step_size=[cfg.c])

        # second half is used for intrinsic reward calculation
        self.goal_buffer.replace_second_half(goals_batch)
        self.st_buffer.replace_second_half(st_batch)

    def store_lstm_states(self):
        # need to split models within session
        self.worker_start_lstm_state = self.session.op_get_lstm_state
        self.manager_start_lstm_state = self.session.op_get_lstm_state

    def get_cosine(self, state):
        z_t = self.session.op_get_zt(ph_state=[state])
        s_t = self.session.self.op_get_st(ph_perception=z_t)[0]

        cur_st = s_t - self.st_buffer.data[-self.cur_c:, :]
        cur_st_norm = \
            np.maximum(np.linalg.norm(cur_st, axis=1), self.eps)
        st_normed = (cur_st.transpose() / cur_st_norm).transpose()

        cur_goal = self.goal_buffer.data[-self.cur_c:, :]
        cur_goal_norm = \
            np.maximum(np.linalg.norm(cur_goal, axis=1), self.eps)
        goals_normed = cur_goal.transpose() / cur_goal_norm

        cosine = np.dot(st_normed, goals_normed)
        return cosine

    def manager_step(self, state):
        zt_inp = self.session.op_get_zt(ph_state=[state])
        self.last_zt_inp, = zt_inp

        goal, m_value, s_t, lstm_state = self.session.op_get_goal_value_st(
            ph_perception=zt_inp,
            ph_initial_lstm_state=self.session.op_get_lstm_state,
            ph_step_size=[1])
        self.session.op_assign_lstm_state(ph_variable=lstm_state)

        self.last_m_value, = m_value
        self.goal_buffer.extend(goal[0])
        self.last_goal = self.goal_buffer.get_sum()
        self.st_buffer.extend(s_t[0])

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
        zt = self.session.op_get_zt(ph_state=[self.states[-1]])
        m_value, _ = self.session.op_get_value(
            ph_perception=zt,
            ph_initial_lstm_state=self.session.op_get_lstm_state,
            ph_step_size=[1])
        self.last_m_value, = m_value

        action, value, _ = self.session.op_get_action_and_value(
            ph_state=[self.states[-1]],
            ph_goal=[self.last_goal],
            ph_initial_lstm_state=self.session.op_get_lstm_state,
            ph_step_size=[1])
        probabilities, = action
        value, = value
        return utils.choose_action(probabilities), value

    def get_discounted_rewards(self, experience):
        r = ri = 0.0
        if self.last_value is not None:
            r = self.last_value
            ri = self.last_m_value

        reward = experience['reward']
        m_reward = experience['reward_i']
        discounted_reward = np.zeros_like(reward, dtype=np.float32)
        m_discounted_reward = np.zeros_like(m_reward, dtype=np.float32)

        # compute discounted rewards
        for t in reversed(range(len(reward))):
            r = reward[t] + cfg.worker_gamma * r
            ri = m_reward[t] + cfg.manager_gamma * ri
            discounted_reward[t] = r
            m_discounted_reward[t] = ri

        return discounted_reward, m_discounted_reward

    def compute_gradients(self, experience):
        discounted_reward, m_discounted_reward = \
            self.get_discounted_rewards(experience)

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
