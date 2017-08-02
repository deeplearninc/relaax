from __future__ import absolute_import

from builtins import range
from builtins import object

import logging
import numpy as np
import scipy.signal
import six.moves.queue as queue
import threading

from relaax.common import profiling
from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from .. import dqn_config
from .. import dqn_model
from . import dqn_utils


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class DQNEpisode(object):
    def __init__(self, parameter_server, metrics, exploit, hogwild_update):
        self.exploit = exploit
        self.ps = parameter_server
        self.metrics = metrics
        model = dqn_model.AgentModel()
        self.session = session.Session(model)
        self.reset()
        self.replay_buffer = dqn_utils.ReplayBuffer(dqn_config.config.replay_buffer_size)
        self.observation = dqn_utils.DQNObservation()
        self.last_action = None
        self.global_step = 0
        if hogwild_update:
            self.queue = queue.Queue(10)
            threading.Thread(target=self.execute_tasks).start()
            self.receive_experience()
        else:
            self.queue = None

    @property
    def experience(self):
        return self.episode.experience

    @profiler.wrap
    def begin(self):
        self.do_task(self.receive_experience)
        self.get_action()
        self.episode.begin()

    @profiler.wrap
    def step(self, reward, state, terminal):
        self.global_step = self.ps.session.op_global_step()
        if self.global_step % 2000:
            self.ps.session.op_update_target_weights()

        self.update()
        self.do_task(self.receive_experience)

        if reward is not None:
            self.push_experience(reward, state, terminal)
        else:
            self.observation.add_state(state)

        if terminal:
            self.observation.add_state(None)

        assert self.last_action is None
        self.get_action()

    @profiler.wrap
    def update(self):
        if self.global_step > dqn_config.config.start_sample_step:
            experience = self.replay_buffer.sample(dqn_config.config.batch_size)
            if not self.exploit:
                self.do_task(lambda: self.send_experience(experience))

    @profiler.wrap
    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action', 'value')

    # Helper methods

    def execute_tasks(self):
        while True:
            task = self.queue.get()
            task()

    def do_task(self, f):
        if self.queue is None:
            f()
        else:
            self.queue.put(f)

    @profiler.wrap
    def send_experience(self, experience):
        batch = dict(zip(experience[0], zip(*[d.values() for d in experience])))
        q_next_target = self.session.get_q_target_value(next_state=batch["next_state"])

        feeds = dict(state=batch["state"],
                     reward=batch["reward"],
                     action=batch["action"],
                     terminal=batch["terminal"],
                     q_next_target=q_next_target)

        if dqn_config.config.double_dqn:
            feeds["q_next"] = self.session.get_q_value(state=batch["next_state"])

        grads = self.session.op_compute_gradients(**feeds)

        self.ps.session.op_apply_gradients(gradients=grads, increment=1)

    @profiler.wrap
    def receive_experience(self):
        weights = self.ps.session.op_get_weights()
        self.session.op_assign_weights(weights=weights)

        target_weights = self.ps.session.op_get_target_weights()
        self.session.op_assign_target_weights(target_weights=target_weights)

    def push_experience(self, reward, state, terminal):
        assert self.observation.queue is not None
        assert self.last_action is not None

        old_state = self.observation.queue
        if state is not None:
            self.observation.add_state(state)

        self.replay_buffer.append(dict(state=old_state,
                                       action=self.last_action,
                                       reward=reward,
                                       terminal=terminal,
                                       next_state=self.observation.queue))

        self.last_action = None

    def get_action(self):
        if self.observation.queue is None:
            self.last_action = None
        else:
            self.last_action = self.get_action_from_network()
            assert self.last_action is not None

    def get_action_from_network(self):
        q_value = self.session.get_q_value(state=[self.observation.queue])

        return self.session.get_action(global_step=self.global_step,
                                       q_value=q_value)
