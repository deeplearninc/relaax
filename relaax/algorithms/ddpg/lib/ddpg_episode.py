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

from .. import ddpg_config
from .. import ddpg_model


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class DDPGEpisode(object):
    def __init__(self, parameter_server, exploit, hogwild_update):
        self.exploit = exploit
        self.ps = parameter_server
        model = ddpg_model.AgentModel()
        self.session = session.Session(model)

        self.reset()
        self.observation = None
        self.last_action = None
        self.last_value = None
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
        self.get_action_and_value()
        self.episode.begin()

    @profiler.wrap
    def step(self, reward, state, terminal):
        if reward is not None:
            self.push_experience(reward)
        if terminal and state is not None:
            logger.warning('DDPGEpisode.step ignores state in case of terminal.')
        else:
            assert (state is None) == terminal
        self.observation.add_state(state)

        self.terminal = terminal
        assert self.last_action is None
        assert self.last_value is None

        self.get_action_and_value()

    @profiler.wrap
    def end(self):
        experience = self.episode.end()
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
