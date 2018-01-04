from __future__ import absolute_import

import time
import logging
import numpy as np
from scipy.signal import lfilter

from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from . import trpo_config
from . import trpo_model
from .lib import network

logger = logging.getLogger(__name__)


class ParameterServer(parameter_server_base.ParameterServerBase):
    def init_session(self):
        self.session = session.Session(trpo_model.SharedParameters())
        if trpo_config.config.async:
            self.session.ps = PsAsync(self.session, self.metrics, self)
        else:
            self.session.ps = Ps(self.session, self.metrics, self)
        self.session.op_initialize()

    def n_step(self):
        return self.session.op_n_step()

    def score(self):
        return self.session.op_score()


class Ps(object):
    def __init__(self, relaax_session, metrics, ps):
        self.relaax_session = relaax_session
        self._metrics = metrics
        self._ps = ps

        self.paths = []             # experience accumulator
        self.paths_len = 0          # length of experience

        self.baseline = network.make_baseline_wrapper(relaax_session.value, metrics)
        self.updater = network.Updater(relaax_session.policy)

        if trpo_config.config.use_filter:
            self.M = np.zeros(trpo_config.config.state_size)
            self.S = np.zeros(trpo_config.config.state_size)

        # Create an update predicate based on config options
        if trpo_config.config.timesteps_per_batch is not None:
            # Update when total number of timesteps is reached
            self.update_condition = self.step_update_condition
        else:
            # Update when number of episodes is reached
            self.update_condition = self.episodic_update_condition

    def episodic_update_condition(self):
        return len(self.paths) >= trpo_config.config.episodes_per_batch

    def step_update_condition(self):
        return self.paths_len >= trpo_config.config.timesteps_per_batch

    def get_global_t(self):
        return self.relaax_session.op_n_step()

    def get_filter_state(self):
        return self.relaax_session.op_n_step(), self.M, self.S

    def wait_for_iteration(self):
        return self.relaax_session.op_n_iter()

    def send_experience(self, n_iter, paths, length):
        if n_iter == self.n_iter():
            self.update_paths(paths, length)

    def receive_weights(self, n_iter):
        assert n_iter == self.n_iter()    # check iteration
        return self._ps.policy_net.get_weights()

    def metrics(self):
        return self._metrics

    def update_paths(self, paths, length):
        self.relaax_session.op_inc_step(increment=length)
        self.paths_len += length
        self.paths.append(paths)

        if trpo_config.config.use_filter:
            self.update_filter_state(paths["filter_diff"])

        if self.update_condition():
            self.trpo_update()
            self.paths_len = 0
            self.paths = []

    def trpo_update(self):
        self.relaax_session.op_turn_collect_off()
        start = time.time()

        self.relaax_session.op_next_iter()
        self.compute_advantage()

        # Value Update
        vf_metrics = self.baseline.fit(self.paths)
        logger.debug("VF metrics: {}".format(vf_metrics))
        for key, value in vf_metrics.items():
            self._metrics.scalar(key, value)

        # Policy Update
        policy_metrics = self.updater(self.paths)
        logger.debug("Policy metrics: {}".format(policy_metrics))
        for key, value in policy_metrics.items():
            self._metrics.scalar(key, value)

        print('Update time for {} iteration: {}'.format(self.n_iter(), time.time() - start))

        # Submit metrics

        self.relaax_session.op_turn_collect_on()

    def compute_advantage(self):
        # Compute & Add to paths: return, baseline, advantage
        for path in self.paths:
            path["return"] = self.discount(path["reward"], trpo_config.config.PG_OPTIONS.rewards_gamma)
            b = path["baseline"] = self.baseline.predict(path)
            b1 = np.append(b, 0 if path["terminated"] else b[-1])
            deltas = path["reward"] + trpo_config.config.PG_OPTIONS.rewards_gamma * b1[1:] - b1[:-1]
            path["advantage"] = self.discount(deltas, trpo_config.config.PG_OPTIONS.rewards_gamma *
                                              trpo_config.config.PG_OPTIONS.gae_lambda)
        alladv = np.concatenate([path["advantage"] for path in self.paths])
        allrwd = np.concatenate([path["reward"] for path in self.paths])
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for path in self.paths:
            path["advantage"] = (path["advantage"] - mean) / std

        self.relaax_session.op_add_reward_to_model_score_routine(reward_sum=np.sum(allrwd),
                                                                 reward_weight=allrwd.shape[0])

    def update_filter_state(self, diff):
        self.M = (self.M*self.relaax_session.op_n_step() + diff[1]) / (
                self.relaax_session.op_n_step() + diff[0])
        self.S += diff[2]

    def n_iter(self):
        return self.relaax_session.op_n_iter_value()

    def discount(self, x, gamma):
        """
        computes discounted sums along 0th dimension of x.

        inputs
        ------
        x: ndarray
        gamma: float

        outputs
        -------
        y: ndarray with same shape as x, satisfying

            y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                    where k = len(x) - t - 1
        """
        x = np.array(x)
        assert x.ndim >= 1
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class PsAsync(Ps):
    def __init__(self, metrics, ps):
        super(PsAsync, self).__init__(metrics, ps)

    def wait_for_iteration(self):
        return self.n_iter()

    def send_experience(self, n_iter, paths, length):
        self.update_paths(paths, length)
