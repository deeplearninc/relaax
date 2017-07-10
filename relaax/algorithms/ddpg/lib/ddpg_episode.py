from __future__ import absolute_import

from builtins import range
from builtins import object

import logging
import numpy as np
import six.moves.queue as queue
import threading

from relaax.common import profiling
from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from .. import ddpg_config as cfg
from .. import ddpg_model
from . import ddpg_observation


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class DDPGEpisode(object):
    def __init__(self, parameter_server, exploit, hogwild_update):
        self.exploit = exploit
        self.ps = parameter_server
        model = ddpg_model.AgentModel()
        self.session = session.Session(model)

        self.episode = episode.ReplayBuffer('state', 'action', 'reward', 'terminal', 'next_state',
                                            buffer_size=cfg.config.buffer_size)
        self.observation = ddpg_observation.DDPGObservation()
        self.last_action = None

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
        if reward is not None:
            self.push_experience(reward, state, terminal)
        else:
            self.observation.add_state(state)

        assert self.last_action is None
        self.get_action()

    @profiler.wrap
    def end(self):
        experience = self.episode.sample(cfg.config.batch_size)
        if not self.exploit:
            self.do_task(lambda: self.send_experience(experience))

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
        self.apply_gradients(self.compute_gradients(experience), len(experience))

    @profiler.wrap
    def receive_experience(self):
        self.session.op_assign_actor_weights(weights=self.ps.session.op_get_actor_weights())
        self.session.op_assign_critic_weights(weights=self.ps.session.op_get_critic_weights())
        self.session.op_assign_actor_target_weights(weights=self.ps.session.op_get_actor_target_weights())
        self.session.op_assign_critic_target_weights(weights=self.ps.session.op_get_critic_target_weights())

    def push_experience(self, reward, state, terminal):
        assert self.observation.queue is not None
        assert self.last_action is not None

        old_state = self.observation.queue
        if state is not None:
            self.observation.add_state(state)

        self.episode.step(
            state=old_state,
            action=self.last_action,
            reward=reward,
            terminal=terminal,
            next_state=self.observation.queue
        )
        self.last_action = None

    def get_action(self):
        if self.observation.queue is None:
            self.last_action = None
        else:
            self.last_action = self.get_action_from_network()
            assert self.last_action is not None

    def get_action_from_network(self):
        out, scaled_out = self.session.op_get_action(state=[self.observation.queue])
        return scaled_out
