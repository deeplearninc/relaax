from __future__ import absolute_import

from builtins import object

import logging

from relaax.common import profiling
from relaax.server.common import session
from relaax.common.algorithms.lib import utils
from relaax.common.algorithms.lib import observation

from .. import dqn_config
from .. import dqn_model
from . import dqn_utils

logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class Trainer(object):
    def __init__(self, parameter_server, metrics, exploit):
        self.ps = parameter_server
        self.metrics = metrics
        self._exploit = exploit

        self.session = session.Session(dqn_model.AgentModel())
        self.session.op_initialize()

        self.replay_buffer = dqn_utils.ReplayBuffer(dqn_config.config.replay_buffer_size,
                                                    dqn_config.config.alpha)
        self.observation = observation.Observation(dqn_config.config.input.history)

        self.last_action = None
        self.local_step = 0
        self.last_target_weights_update = 0

    @profiler.wrap
    def begin(self):
        self.get_action()

    @profiler.wrap
    def step(self, reward, state, terminal):
        self.local_step += 1

        if self.local_step % dqn_config.config.update_target_weights_interval == 0:
            self.session.op_update_target_weights()

        self.receive_experience()

        if self.local_step > dqn_config.config.start_sample_step:
            self.update()

        # metrics
        if state is not None:
            self.metrics.histogram('state', state)

        if reward is None:
            self.observation.add_state(state)
        else:
            self.push_experience(reward, state, terminal)

        if terminal:
            self.observation.add_state(None)

        assert self.last_action is None
        self.get_action()

    @profiler.wrap
    def update(self):
        experience = self.replay_buffer.sample(dqn_config.config.batch_size)
        self.send_experience(experience)

    @profiler.wrap
    def send_experience(self, experience):
        batch = dict(zip(experience[0], zip(*[d.values() for d in experience])))
        q_next_target = self.session.op_get_q_target_value(next_state=batch["next_state"])
        q_next = self.session.op_get_q_value(state=batch["next_state"])

        feeds = dict(state=batch["state"],
                     reward=batch["reward"],
                     action=batch["action"],
                     terminal=batch["terminal"],
                     q_next_target=q_next_target,
                     q_next=q_next)

        gradients = self.session.op_compute_gradients(**feeds)

        for i, g in enumerate(utils.Utils.flatten(gradients)):
            self.metrics.histogram('gradients_%d' % i, g)

        self.ps.session.op_apply_gradients(gradients=gradients)

    @profiler.wrap
    def receive_experience(self):
        weights = self.ps.session.op_get_weights()
        self.session.op_assign_weights(weights=weights)

    def push_experience(self, reward, state, terminal):
        assert not self.observation.is_none()
        assert self.last_action is not None

        old_state = self.observation.get_state()
        if state is not None:
            self.observation.add_state(state)

        self.replay_buffer.append(dict(state=old_state,
                                       action=self.last_action,
                                       reward=reward,
                                       terminal=terminal,
                                       next_state=self.observation.get_state()))

        self.last_action = None

    def get_action(self):
        if self.observation.is_none():
            self.last_action = None
        else:
            q_value = self.session.op_get_q_value(state=[self.observation.get_state()])
            self.last_action = self.session.op_get_action(local_step=self.local_step, q_value=q_value)

            assert self.last_action is not None

            # metrics
            self.metrics.histogram('action', self.last_action)
