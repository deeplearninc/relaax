from __future__ import print_function

import numpy as np
from scipy.signal import lfilter
from time import time

from ... import trpo_config

from ..common import network


class ParameterServer(object):
    def __init__(self, saver_factory, metrics, relaax_session):
        self.relaax_session = relaax_session

        self.paths = []             # experience accumulator
        self.paths_len = 0          # length of experience

        self.baseline = network.make_baseline_wrapper(relaax_session, metrics)
        self.trpo_updater = network.TrpoUpdater(relaax_session)

        self._saver = None

        if trpo_config.config.use_filter:
            self.M = np.zeros(trpo_config.config.state_size)
            self.S = np.zeros(trpo_config.config.state_size)

        self._bridge = _Bridge(metrics, self)
        if trpo_config.config.async:
            self._bridge = _BridgeAsync(metrics, self)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        checkpoint_ids = self._saver.checkpoint_ids()
        if len(checkpoint_ids) > 0:
            self._saver.restore_checkpoint(max(checkpoint_ids))

    def save_checkpoint(self):
        self._saver.save_checkpoint((self.n_iter(), self.paths_len))

    def checkpoint_location(self):
        return self._saver.location()

    def bridge(self):
        return self._bridge

    def update_paths(self, paths, length):
        self.relaax_session.op_inc_step(increment=length)
        self.paths_len += length
        self.paths.append(paths)

        if trpo_config.config.use_filter:
            self.update_filter_state(paths["filter_diff"])

        #if self.paths_len >= trpo_config.config.PG_OPTIONS.timesteps_per_batch:
        if len(self.paths) >= trpo_config.config.batch_size:
            self.trpo_update()
            self.paths_len = 0
            self.paths = []

    def trpo_update(self):
        self.relaax_session.op_turn_collect_off()
        start = time()

        self.relaax_session.op_next_iter()
        self.compute_advantage()
        # Value Update
        self.baseline.fit(self.paths)
        # Policy Update
        self.trpo_updater(self.paths)

        print('Update time for {} iteration: {}'.format(self.n_iter(), time() - start))
        self.relaax_session.op_turn_collect_on()

    def compute_advantage(self):
        # Compute & Add to paths: return, baseline, advantage
        for path in self.paths:
            path["return"] = discount(path["reward"], trpo_config.config.PG_OPTIONS.rewards_gamma)
            b = path["baseline"] = self.baseline.predict(path)
            b1 = np.append(b, 0 if path["terminated"] else b[-1])
            deltas = path["reward"] + trpo_config.config.PG_OPTIONS.rewards_gamma * b1[1:] - b1[:-1]
            path["advantage"] = discount(deltas, trpo_config.config.PG_OPTIONS.rewards_gamma *
                                         trpo_config.config.PG_OPTIONS.gae_lambda)
        alladv = np.concatenate([path["advantage"] for path in self.paths])
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for path in self.paths:
            path["advantage"] = (path["advantage"] - mean) / std

    def global_t(self):
        return self.relaax_session.op_n_step()

    def filter_state(self):
        return self.relaax_session.op_n_step(), self.M, self.S

    def update_filter_state(self, diff):
        self.M = (self.M*self.relaax_session.op_n_step() + diff[1]) / (
                self.relaax_session.op_n_step() + diff[0])
        self.S += diff[2]

    def n_iter(self):
        return self.relaax_session.op_n_iter_value()


class _Bridge(object):
    def __init__(self, metrics, ps):
        self._metrics = metrics
        self._ps = ps

    def get_global_t(self):
        return self._ps.global_t()

    def get_filter_state(self):
        return self._ps.filter_state()

    def wait_for_iteration(self):
        return self._ps.relaax_session.op_n_iter()

    def send_experience(self, n_iter, paths, length):
        if n_iter == self._ps.n_iter():
            self._ps.update_paths(paths, length)

    def receive_weights(self, n_iter):
        assert n_iter == self._ps.n_iter()    # check iteration
        return self._ps.policy_net.get_weights()

    def metrics(self):
        return self._metrics


class _BridgeAsync(_Bridge):
    def __init__(self, metrics, ps):
        super(_BridgeAsync, self).__init__(metrics, ps)

    def wait_for_iteration(self):
        return self._ps.n_iter()

    def send_experience(self, n_iter, paths, length):
        self._ps.update_paths(paths, length)


def discount(x, gamma):
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
