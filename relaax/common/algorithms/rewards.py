from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import moves

import tensorflow as tf
import numpy as np


def discounted_reward(rewards, gamma=0.99, normalize=True, constraint=False):
    """Computes discounted sums of rewards along 0-th dimension.

    y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
            where k = len(x) - t - 1

    Args:
        rewards (list or numpy.ndarray): List of rewards to process.
        gamma (float): Discount factor.
        normalize (bool): if True -> additionally normalize the computed rewards.
        constraint (bool): if True -> add unity boundary constraint when computing.

    Returns:
        discounted rewards (numpy.ndarray): has the same shape as input Args:rewards.

    Examples
    ----------
    >>> rewards = [1, 0, 0, 1, 0,
    >>>            0, 0, 1, 1, 1]
    >>> gamma = 0.95
    >>> print(discounted_reward(rewards, gamma, normalize=False, constraint=True))
    ... [ 1.      0.9025    0.95    1.    0.85737503
    ...   0.9025  0.95      1.      1.    1.        ]
    """
    # initialize resulting array for discounted rewards & running accumulator
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0

    for t in reversed(moves.xrange(0, discounted_r.size)):
        if constraint and rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    # size the rewards to be unit normal (helps control the gradient estimator variance)
    if normalize:
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r) + 1e-20
    return discounted_r


def get_gamma_operator(batch_size_, gamma_=0.99):
    """Computes gamma_operator wrt batch (buffer) size & gamma.

    Args:
        batch_size_ (int): The upper limit of the powers to compute.
        gamma_ (float): Discount factor.

    Returns:
        gamma_operator (numpy.ndarray): triangle matrix with powers of gamma.

    Examples
    ----------
    >>> batch_size = 3
    >>> gamma = 0.95
    >>> print(get_gamma_operator(batch_size, gamma))
    >>> [[1.    0.95    0.9025]
    >>>  [0.    1.      0.95  ]
    >>>  [0.    0.      1.    ]
    """
    # create array of gamma's powers from [0..batch_size)
    powers = np.power(np.ones(batch_size_) * gamma_, np.arange(batch_size_))

    # create & fill upper triangle ladder matrix
    gamma_operator = np.zeros((batch_size_, batch_size_))
    for idx in moves.xrange(batch_size_):
        gamma_operator[idx, idx:] = powers[:batch_size_ - idx]

    return gamma_operator


class DiscountedReward(object):
    def __init__(self, gamma_operator_):
        # initialize tensorflow variable with incoming upper triangle ladder matrix
        gamma_operator = tf.Variable(gamma_operator_, name='gamma_operator', dtype=np.float32)

        # define interfaces to feed in: rewards array and its length
        self._ph_rewards = tf.placeholder(tf.float32, [None, 1], name='ph_rewards')
        self._length = tf.placeholder(tf.int32, [1], name='slice_index')

        # operation to compute a column vector of discounted rewards by appropriate matrix multiplication
        self._compute = tf.matmul(gamma_operator[:self._length, :self._length], self._ph_rewards)

        # additional operations for normalization the resulting rewards wrt its mean and std
        mean_centered = self._compute - tf.reduce_mean(self._compute)
        epsilon = tf.constant(1e-20)  # to prevent zero dividing when all rewards are equal to 0
        self._normalize = mean_centered / tf.sqrt(tf.nn.moments(mean_centered, axes=[0])[1] + epsilon)

    def __call__(self, sess, rewards, normalize=False):
        """Returns computed discounted rewards as a column vector.

        Args:
            sess : Tensorflow session.
            rewards (numpy.ndarray): Column vector with shape == (batch_size, 1).
            normalize (bool): if True -> additionally normalize the computed rewards.

        Returns:
            discounted rewards (numpy.ndarray): Has the same shape as input Args:rewards.
        """

        # dictionary to pass in respective tensorflow op, where we additionally
        # retrieves the batch_size from the 1st dimension of incoming column vector
        feeds = {self._ph_rewards: rewards, self._length: rewards.shape[0]}

        if normalize:
            return sess.run(self._normalize, feed_dict=feeds)
        return sess.run(self._compute, feed_dict=feeds)


class DiscountedRewardFull(object):
    """Take numpy (column) array of rewards and compute discounted rewards.

    Examples
    ----------
    >>> rewards = [1, 0, 0, 1, 0,
    >>>            0, 0, 1, 1, 1]
    >>> gamma = 0.95
    >>> dr_caller = DiscountedRewardFull(len(rewards), gamma)
    >>> with tf.Session() as sess:
    >>>    sess.run(tf.variables_initializer(tf.global_variables()))
    >>>    # we have to rebuild rewards list to numpy column vector with `np.vstack`
    >>>    discounted_rewards = dr_caller(sess, np.vstack(rewards))
    >>>    print(discounted_rewards)
    ... [[ 3.84938192]
    ...  [ 2.99934936]
    ...  [ 3.15720987]
    ...  [ 3.3233788 ]
    ...  [ 2.44566202]
    ...  [ 2.57438111]
    ...  [ 2.70987487]
    ...  [ 2.85249996]
    ...  [ 1.95000005]
    ...  [ 1.        ]]
    """
    def __init__(self, batch_size_, gamma_=0.99):
        """Compute gamma operator wrt batch (buffer) size & define ops.

        Args:
            batch_size_ (int): The upper limit of the powers to compute.
            gamma_ (float): Discount factor.
        """

        # create array of gamma's powers from [0..batch_size)
        powers = np.power(np.ones(batch_size_) * gamma_, np.arange(batch_size_))

        """ create & fill upper triangle ladder matrix
        Example
        ----------
        >>> # batch_size_ == 3
        >>> [[1.    0.95    0.9025]
        >>>  [0.    1.      0.95  ]
        >>>  [0.    0.      1.    ]]
        """
        init = np.zeros((batch_size_, batch_size_))
        for idx in moves.xrange(batch_size_):
            init[idx, idx:] = powers[:batch_size_ - idx]

        # initialize tensorflow variable with computed upper triangle ladder matrix
        gamma_operator = tf.Variable(init, name='gamma_operator', dtype=np.float32)

        # define interfaces to feed in: rewards array and its length
        self._ph_rewards = tf.placeholder(tf.float32, [None, 1], name='ph_rewards')
        self._length = tf.placeholder(tf.int32, [1], name='slice_index')

        # operation to compute a column vector of discounted rewards by appropriate matrix multiplication
        self._compute = tf.matmul(gamma_operator[:self._length, :self._length], self._ph_rewards)

        # additional operations for normalization the resulting rewards wrt its mean and std
        mean_centered = self._compute - tf.reduce_mean(self._compute)
        epsilon = tf.constant(1e-20)    # to prevent zero dividing when all rewards are equal to 0
        self._normalize = mean_centered / tf.sqrt(tf.nn.moments(mean_centered, axes=[0])[1] + epsilon)

    def __call__(self, sess, rewards, normalize=False):
        """Returns computed discounted rewards as a column vector.

        Args:
            sess : Tensorflow session.
            rewards (numpy.ndarray): Column vector with shape == (batch_size, 1).
            normalize (bool): if True -> additionally normalize the computed rewards.

        Returns:
            discounted rewards (numpy.ndarray): Has the same shape as input Args:rewards.
        """

        # dictionary to pass in respective tensorflow op, where we additionally
        # retrieves the batch_size from the 1st dimension of incoming column vector
        feeds = {self._ph_rewards: rewards, self._length: rewards.shape[0]}

        if normalize:
            return sess.run(self._normalize, feed_dict=feeds)
        return sess.run(self._compute, feed_dict=feeds)
