from . import *


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
