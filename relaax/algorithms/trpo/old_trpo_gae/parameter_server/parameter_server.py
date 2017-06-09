from __future__ import print_function

import keras.backend
import os.path
import re
import tensorflow as tf
import numpy as np
from scipy.signal import lfilter
from time import time

from . import network


class ParameterServer(object):
    def __init__(self, config, saver_factory, metrics):
        self.n_iter = 0             # number of updates within training process
        self.config = config        # common configuration, which is rewritten by yaml

        self.is_collect = True      # set to False if TRPO is under update procedure (for sync only)
        self.paths = []             # experience accumulator
        self.paths_len = 0          # length of experience
        self.global_step = 0        # step accumulator of whole experience through all updates

        # inform Keras that we are going to initialize variables here
        keras.backend.manual_variable_initialization(True)

        self._session = tf.Session()
        keras.backend.set_session(self._session)

        self.policy_net, self.value_net = network.make(config)

        self.policy, self.baseline = network.make_head(config, self.policy_net, self.value_net, self._session)
        self.trpo_updater = network.make_trpo(config, self.policy, self._session)

        self._saver = None

        self._session.run(tf.variables_initializer(tf.global_variables()))

        if config.use_filter:
            self.M = np.zeros(config.state_size)
            self.S = np.zeros(config.state_size)

        self._bridge = _Bridge(metrics, self)
        if config.async:
            self._bridge = _BridgeAsync(metrics, self)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        checkpoint_ids = self._saver.checkpoint_ids()
        if len(checkpoint_ids) > 0:
            self._saver.restore_checkpoint(max(checkpoint_ids))

    def save_checkpoint(self):
        self._saver.save_checkpoint((self.n_iter, self.paths_len))

    def checkpoint_location(self):
        return self._saver.location()

    def bridge(self):
        return self._bridge

    def update_paths(self, paths, length):
        self.global_step += length
        self.paths_len += length
        self.paths.append(paths)

        if self.config.use_filter:
            self.update_filter_state(paths["filter_diff"])

        if self.paths_len >= self.config.PG_OPTIONS.timesteps_per_batch:
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

        print('Update time for {} iteration: {}'.format(self.n_iter, time() - start))
        self.is_collect = True

    def compute_advantage(self):
        # Compute & Add to paths: return, baseline, advantage
        for path in self.paths:
            path["return"] = discount(path["reward"], self.config.PG_OPTIONS.rewards_gamma)
            b = path["baseline"] = self.baseline.predict(path)
            b1 = np.append(b, 0 if path["terminated"] else b[-1])
            deltas = path["reward"] + self.config.PG_OPTIONS.rewards_gamma * b1[1:] - b1[:-1]
            path["advantage"] = discount(deltas, self.config.PG_OPTIONS.rewards_gamma * self.config.PG_OPTIONS.gae_lambda)
        alladv = np.concatenate([path["advantage"] for path in self.paths])
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for path in self.paths:
            path["advantage"] = (path["advantage"] - mean) / std

    def global_t(self):
        return self.global_step

    def filter_state(self):
        return self.global_step, self.M, self.S

    def update_filter_state(self, diff):
        self.M = (self.M*self.global_step + diff[1]) / (self.global_step + diff[0])
        self.S += diff[2]


class _Bridge(object):
    def __init__(self, metrics, ps):
        self._metrics = metrics
        self._ps = ps

    def get_global_t(self):
        return self._ps.global_t()

    def get_filter_state(self):
        return self._ps.filter_state()

    def wait_for_iteration(self):
        if self._ps.is_collect:
            return self._ps.n_iter
        return -1

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


class _Checkpoint(relaax.server.common.saver.checkpoint.Checkpoint):
    _PNET_S = 'pnet-%d-%d.h5'
    _PNET_RE = re.compile('^pnet-(\d+)-(\d+).h5$')

    _VNET_S = 'vnet-%d-%d.h5'
    _VNET_RE = re.compile('^vnet-(\d+)-(\d+).h5$')

    _DATA_S = 'data-%d-%d.p'
    _DATA_RE = re.compile('^data-(\d+)-(\d+).p$')

    def __init__(self, ps):
        self._ps = ps

    def checkpoint_ids(self, names):
        ids = set()
        for name in names:
            match = self._DATA_RE.match(name)
            if match is not None:
                ids.add((int(match.group(1)), int(match.group(2))))
        return ids

    def checkpoint_names(self, names, checkpoint_id):
        items = (
            self._PNET_S % checkpoint_id,
            self._VNET_S % checkpoint_id,
            self._DATA_S % checkpoint_id
        )
        return tuple(set(names) & set(items))

    def restore_checkpoint(self, dir, checkpoint_id):
        self._ps.policy_net.load_weights(os.path.join(dir, self._PNET_S % checkpoint_id))
        self._ps.value_net.load_weights(os.path.join(dir, self._VNET_S % checkpoint_id))
        with open(os.path.join(dir, self._DATA_S % checkpoint_id), 'rb') as f:
            if self._ps.config.use_filter:
                self._ps.paths, self._ps.global_step, self._ps.M, self._ps.S = load(f)
            else:
                self._ps.paths, self._ps.global_step = load(f)

    def save_checkpoint(self, dir, checkpoint_id):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self._ps.policy_net.save_weights(os.path.join(dir, self._PNET_S % checkpoint_id))
        self._ps.value_net.save_weights(os.path.join(dir, self._VNET_S % checkpoint_id))
        with open(os.path.join(dir, self._DATA_S % checkpoint_id), 'wb') as f:
            if self._ps.config.use_filter:
                dump((self._ps.paths[:], self._ps.global_step, self._ps.M, self._ps.S), f)
            else:
                dump((self._ps.paths[:], self._ps.global_step), f)


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
