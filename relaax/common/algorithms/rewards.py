from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import moves
import tensorflow as tf
import numpy as np


class DiscountedReward(object):
    """Take numpy (column) array of rewards and compute discounted rewards.

    Examples
    ----------
    >>> rewards = [1, 0, 0, 1, 0,
    >>>            0, 0, 1, 1, 1]
    >>> gamma = 0.95
    >>> dr_caller = DiscountedReward(gamma, len(rewards))
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
    def __init__(self, gamma_, batch_size_):
        """Compute gamma operator wrt batch (buffer) size & define ops.

        Args:
            gamma_ (float): Discount factor.
            batch_size_ (int): The upper limit of the powers to compute.
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
        self._normalize = mean_centered / tf.sqrt(tf.nn.moments(mean_centered, axes=[0])[1])

    def __call__(self, sess, rewards, normalize=False):
        """Returns computed discounted rewards as column vector.

        Args:
            sess : Tensorflow session.
            rewards (numpy array): column vector with shape == (batch_size, 1).
            normalize (bool): if True -> additionally normalize the computed rewards.

        Returns:
            discounted rewards (numpy array): has the same shape as input Args:rewards.
        """

        # dictionary to pass in respective tensorflow op, where we additionally
        # retrieves the batch_size from the 1st dimension of incoming column vector
        feeds = {self._ph_rewards: rewards, self._length: rewards.shape[0]}

        if normalize:
            return sess.run(self._normalize, feed_dict=feeds)
        return sess.run(self._compute, feed_dict=feeds)
