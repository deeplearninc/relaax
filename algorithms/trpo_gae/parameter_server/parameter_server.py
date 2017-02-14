from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy.signal import lfilter
from time import time, sleep

import relaax.algorithm_base.parameter_server_base

from . import network
from saver import KerasSaver as Saver
from cPickle import load    # ujson


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, config, saver, metrics):
        self.n_iter = 0             # number of updates within training process
        self.config = config        # common configuration, which is rewritten by yaml
        self._saver = Saver(saver._savers[0]._dir)  # saver for defined neural networks

        self.is_collect = True      # set to False if TRPO is under update procedure (for sync only)
        self.paths = []             # experience accumulator
        self.paths_len = 0          # length of experience
        self.global_step = 0        # step accumulator of whole experience through all updates

        self._session = tf.Session()

        self.policy_net, self.value_net = network.make(config, self._session)

        initialize = tf.variables_initializer(tf.global_variables())

        self.policy, self.baseline = network.make_head(config, self.policy_net, self.value_net, self._session)
        self.trpo_updater = network.make_trpo(config, self.policy, self._session)

        self._session.run(initialize)

        self._bridge = _Bridge(metrics, self)
        if config.async_collect:
            self._bridge = _BridgeAsync(metrics, self)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        status, self.n_iter, self.paths_len = self._saver.latest_checkpoint_idx()
        print('n_iter =', self.n_iter)
        if status:
            self.policy_net.load_weights(self._saver.dir + "/pnet--" + str(self.n_iter) + ".h5")
            self.value_net.load_weights(self._saver.dir + "/vnet--" + str(self.n_iter) + ".h5")
            self.paths = load(open(self._saver.dir + "/data--" + str(self.n_iter) + "-" + str(self.paths_len) + ".p"))
            self.global_step = (self.n_iter + 1) * self.config.timesteps_per_batch + self.paths_len
        return status

    def save_checkpoint(self):
        self._saver.save_checkpoint(self.policy_net, self.value_net, self.n_iter, self.paths[:], self.paths_len)

    def checkpoint_location(self):
        return self._saver.location()

    def bridge(self):
        return self._bridge

    def update_paths(self, paths, length):
        self.global_step += length
        self.paths_len += length
        self.paths.append(paths)

        if self.paths_len >= self.config.timesteps_per_batch:
            self.trpo_update()
            self.paths_len = 0
            self.paths = []

    def trpo_update(self):
        self.is_collect = False
        start = time()

        self.n_iter += 1
        self.compute_advantage()
        # Value Update
        vf_stats = self.baseline.fit(self.paths)
        # Policy Update
        pol_stats = self.trpo_updater(self.paths)

        print('Update time:', time() - start)
        self.is_collect = True

    def compute_advantage(self):
        # Compute & Add to paths: return, baseline, advantage
        for path in self.paths:
            path["return"] = discount(path["reward"], self.config.GAMMA)
            b = path["baseline"] = self.baseline.predict(path)
            b1 = np.append(b, 0 if path["terminated"] else b[-1])
            deltas = path["reward"] + self.config.GAMMA * b1[1:] - b1[:-1]
            path["advantage"] = discount(deltas, self.config.GAMMA * self.config.LAMBDA)
        alladv = np.concatenate([path["advantage"] for path in self.paths])
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for path in self.paths:
            path["advantage"] = (path["advantage"] - mean) / std

    def global_t(self):
        return self.global_step


class _Bridge(object):
    def __init__(self, metrics, ps):
        self._metrics = metrics
        self._ps = ps

    def wait_for_iteration(self):
        while not self._ps.is_collect:
            sleep(1)
        return self._ps.n_iter

    def send_experience(self, n_iter, paths, length):
        if n_iter == self._ps.n_iter:
            self._ps.update_paths(paths, length)

    def receive_weights(self, n_iter):
        assert n_iter == self._ps.n_iter    # check iteration
        return self._ps.policy_net.get_weights()

    def metrics(self):
        return self._metrics


class _BridgeAsync(_Bridge):
    def __init__(self, metrics, ps):
        super(_BridgeAsync, self).__init__(metrics, ps)

    def wait_for_iteration(self):
        return self._ps.n_iter

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
    assert x.ndim >= 1
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
