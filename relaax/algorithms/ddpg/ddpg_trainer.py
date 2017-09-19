from __future__ import absolute_import
from builtins import object

import logging
import numpy as np
import six.moves.queue as queue
import threading

from relaax.common import profiling
from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import observation
from relaax.common.algorithms.lib import utils

from . import ddpg_config as cfg
from . import ddpg_model


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class Trainer(object):
    def __init__(self, parameter_server, metrics, exploit, hogwild_update):
        self.exploit = exploit
        self.ps = parameter_server
        self.metrics = metrics
        model = ddpg_model.AgentModel()
        self.session = session.Session(model)

        self.episode = episode.ReplayBuffer(cfg.config.buffer_size,
                                            cfg.config.exploration.rnd_seed,
                                            'state', 'action', 'reward', 'terminal', 'next_state')
        self.episode.begin()
        self.observation = observation.Observation(cfg.config.input.history)
        self.last_action = self.noise_epsilon = None
        self.episode_cnt = self.cur_loop_cnt = 0
        self.exploration_noise = utils.OUNoise(cfg.config.output.action_size,
                                               cfg.config.exploration.ou_mu,
                                               cfg.config.exploration.ou_theta,
                                               cfg.config.exploration.ou_sigma,
                                               cfg.config.exploration.rnd_seed)
        self.max_q = self.step_cnt = 0
        self.terminal = False

        if hogwild_update:
            self.queue = queue.Queue(10)
            threading.Thread(target=self.execute_tasks).start()
            self.receive_experience()
        else:
            self.queue = None

        if cfg.config.no_ps:
            self.session.op_initialize()
            self.session.op_init_target_weights()

    @property
    def experience(self):
        return self.episode.experience

    @profiler.wrap
    def begin(self):
        self.do_task(self.receive_experience)
        self.terminal = False
        self.episode_cnt = self.ps.session.op_get_episode_cnt()
        self.exploration_noise.reset(self.episode_cnt + cfg.config.exploration.rnd_seed)
        self.noise_epsilon = np.exp(-self.episode_cnt / cfg.config.exploration.tau)
        self.get_action()

    @profiler.wrap
    def step(self, reward, state, terminal):
        self.step_cnt += 1
        if self.cur_loop_cnt == cfg.config.loop_size:
            self.update()
            self.do_task(self.receive_experience)
            self.cur_loop_cnt = 0

        if reward is not None:
            self.push_experience(reward, state, terminal)
        else:
            self.observation.add_state(state)

        self.cur_loop_cnt += 1
        self.terminal = terminal

        if terminal:
            self.update()
            self.cur_loop_cnt = 0
            self.ps.session.op_inc_episode_cnt(increment=1)
            self.observation.add_state(None)
            Qmax = self.max_q / float(self.step_cnt)
            print('Qmax: %.4f' % Qmax)
            self.metrics.scalar('Qmax', Qmax, self.episode_cnt)
            self.max_q = self.step_cnt = 0

        assert self.last_action is None
        self.get_action()

    @profiler.wrap
    def update(self):
        if self.episode.experience._len > cfg.config.batch_size:
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
        # Calculate targets
        action_target_scaled = self.session.op_get_actor_target(state=experience['next_state'])
        target_q = self.session.op_get_critic_target(state=experience['next_state'],
                                                     action=action_target_scaled.astype(np.float32))

        y = np.asarray(experience['reward']) + \
            cfg.config.rewards_gamma * np.squeeze(target_q) * (~np.asarray(experience['terminal']))

        critic_grads = self.session.op_compute_critic_gradients(state=experience['state'],
                                                                action=experience['action'],
                                                                predicted=np.vstack(y))

        predicted_q = self.session.op_get_critic_q(state=experience['state'],
                                                   action=experience['action'])
        self.max_q += np.amax(predicted_q)

        scaled_out = self.session.op_get_action(state=experience['state'])
        action_grads = self.session.op_compute_critic_action_gradients(state=experience['state'],
                                                                       action=scaled_out)

        actor_grads = self.session.op_compute_actor_gradients(state=experience['state'],
                                                              grad_ys=action_grads)

        if self.terminal and cfg.config.log_lvl == 'DEBUG':
            x = self.episode_cnt
            self.metrics.histogram('action_target_scaled', action_target_scaled, x)

            critic_sq_loss = self.session.op_critic_loss(state=experience['state'],
                                                         action=experience['action'],
                                                         predicted=np.vstack(y))
            self.metrics.histogram('y', y, x)
            self.metrics.scalar('critic_sq_loss', critic_sq_loss, x)

            self.metrics.histogram('q_target', target_q, x)
            self.metrics.histogram('q_predicted', predicted_q, x)

            for i, g in enumerate(utils.Utils.flatten(critic_grads)):
                self.metrics.histogram('grads_critic_%d' % i, g, x)
            for i, g in enumerate(utils.Utils.flatten(action_grads)):
                self.metrics.histogram('grads_action_%d' % i, g, x)
            for i, g in enumerate(utils.Utils.flatten(actor_grads)):
                self.metrics.histogram('grads_actor_%d' % i, g, x)

        if cfg.config.log_lvl == 'VERBOSE':
            norm_critic_grads = self.session.op_compute_norm_critic_gradients(state=experience['state'],
                                                                              action=experience['action'],
                                                                              predicted=np.vstack(y))
            norm_action_grads = self.session.op_compute_norm_critic_action_gradients(state=experience['state'],
                                                                                     action=scaled_out)
            norm_actor_grads = self.session.op_compute_norm_actor_gradients(state=experience['state'],
                                                                            grad_ys=action_grads)
            self.metrics.scalar('grads_norm_critic', norm_critic_grads)
            self.metrics.scalar('grads_norm_action', norm_action_grads)
            self.metrics.scalar('grads_norm_actor', norm_actor_grads)

            self.metrics.histogram('batch_states', experience['state'])
            self.metrics.histogram('batch_actions', experience['action'])
            self.metrics.histogram('batch_rewards', experience['reward'])
            self.metrics.histogram('batch_next_states', experience['next_state'])

        if not cfg.config.no_ps:
            self.ps.session.op_apply_actor_gradients(gradients=actor_grads, increment=self.cur_loop_cnt)
            self.ps.session.op_apply_critic_gradients(gradients=critic_grads)

            self.ps.session.op_update_target_weights()
        else:
            self.session.op_apply_actor_gradients(gradients=actor_grads)
            self.session.op_apply_critic_gradients(gradients=critic_grads)

            self.ps.session.op_inc_step(increment=self.cur_loop_cnt)
            self.session.op_update_target_weights()

    @profiler.wrap
    def receive_experience(self):
        if not cfg.config.no_ps:
            actor_weights, actor_target_weights, critic_weights, critic_target_weights = \
                self.ps.session.op_get_weights()
        else:
            actor_weights, actor_target_weights, critic_weights, critic_target_weights = \
                self.session.op_get_weights()

        self.session.op_assign_actor_weights(weights=actor_weights)
        self.session.op_assign_critic_weights(weights=critic_weights)
        self.session.op_assign_actor_target_weights(weights=actor_target_weights)
        self.session.op_assign_critic_target_weights(weights=critic_target_weights)

        if self.terminal and cfg.config.log_lvl == 'DEBUG':
            x = self.episode_cnt

            for i, g in enumerate(utils.Utils.flatten(actor_weights)):
                self.metrics.histogram('weights_actor_%d' % i, g, x)
            for i, g in enumerate(utils.Utils.flatten(critic_weights)):
                self.metrics.histogram('weights_critic_%d' % i, g, x)

            for i, g in enumerate(utils.Utils.flatten(actor_target_weights)):
                self.metrics.histogram('weights_actor_target_%d' % i, g, x)
            for i, g in enumerate(utils.Utils.flatten(critic_target_weights)):
                self.metrics.histogram('weights_critic_target_%d' % i, g, x)

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
        if cfg.config.log_lvl == 'VERBOSE':
            self.metrics.histogram('step_state', old_state)
            self.metrics.histogram('step_action', self.last_action)
            self.metrics.scalar('step_reward', reward)

        self.last_action = None

    def get_action(self):
        if self.observation.queue is None:
            self.last_action = None
        else:
            self.last_action = self.get_action_from_network()
            if cfg.config.ou_noise:
                self.last_action += self.noise_epsilon * self.exploration_noise.noise()
            else:
                self.last_action += 1. / (1. + float(self.episode_cnt))
            assert self.last_action is not None

    def get_action_from_network(self):
        return self.session.op_get_action(state=[self.observation.queue])[0]
