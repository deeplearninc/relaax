from __future__ import absolute_import

from builtins import object

import logging

from relaax.common import profiling
from relaax.server.common import session
from relaax.common.algorithms.lib import utils

from .. import dqn_config
from .. import dqn_model
from . import dqn_utils


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class DQNEpisode(object):
    def __init__(self, parameter_server, metrics, exploit, hogwild_update):
        self.exploit = exploit  # False
        self.ps = parameter_server
        self.metrics = metrics
        self.session = session.Session(dqn_model.AgentModel())

        # self.session.op_initialize()
        self.replay_buffer = dqn_utils.ReplayBuffer(dqn_config.config.replay_buffer_size)
        self.observation = dqn_utils.DQNObservation()

        self.last_action = None
        self.local_step = 0

        self.queue = None

    @profiler.wrap
    def begin(self):
        # self.do_task(self.receive_experience)
        self.session.op_initialize()
        self.get_action()

    @profiler.wrap
    def step(self, reward, state, terminal):
        self.local_step += 1

        if self.local_step % dqn_config.config.update_target_interval == 0:
            # self.ps.session.op_update_target_weights()
            self.session.op_update_target_weights()
            # print("STEP: %i" % self.local_step)

        if self.local_step > dqn_config.config.start_sample_step:
            self.update()
            # self.do_task(self.receive_experience)

        # metrics
        if state is not None:
            self.metrics.histogram('state', state)

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
        experience = self.replay_buffer.sample(dqn_config.config.batch_size)
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
        batch = dict(zip(experience[0], zip(*[d.values() for d in experience])))  # list of dicts to dict of lists
        q_next_target = self.session.op_get_q_target_value(next_state=batch["next_state"])
        q_next = self.session.op_get_q_value(state=batch["next_state"])

        feeds = dict(state=batch["state"],
                     reward=batch["reward"],
                     action=batch["action"],
                     terminal=batch["terminal"],
                     q_next_target=q_next_target,
                     q_next=q_next)

        # print("\n")
        # print("state: %s" % str(batch["state"]))
        # print("reward: %s" % str(batch["reward"]))
        # if self.local_step > 5000 and self.local_step % 100 == 0:
        #     print("action: %i" % sum(batch["action"]))
        # print("terminal: %s" % str(batch["terminal"]))
        # print("next_state: %s" % str(batch["next_state"]))
        # print("q_next_target: %s" % str(q_next_target))
        # print("q_next: %s" % str(q_next))
        # print("\n")

        # if dqn_config.config.double_dqn:
        #     feeds["q_next"] = self.session.op_get_q_value(state=batch["next_state"])

        gradients = self.session.op_compute_gradients(**feeds)

        for i, g in enumerate(utils.Utils.flatten(gradients)):
            self.metrics.histogram('gradients_%d' % i, g)

        # self.ps.session.op_apply_gradients(gradients=gradients, increment=1)
        self.session.op_apply_gradients(gradients=gradients)

    @profiler.wrap
    def receive_experience(self):
        weights = self.ps.session.op_get_weights()

        # metrics
        for i, w in enumerate(utils.Utils.flatten(weights)):
            self.metrics.histogram('weight_%d' % i, w)

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

            # metrics
            self.metrics.histogram('action', self.last_action)

            assert self.last_action is not None

    def get_action_from_network(self):
        q_value = self.session.op_get_q_value(state=[self.observation.queue])

        return self.session.op_get_action(local_step=self.local_step,
                                          q_value=q_value)
