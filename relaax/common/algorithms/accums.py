from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .acts import choose_action


def accumulate(obj, state, reward, probs):
    """Accumulate experience wrt state, actions and reward for agent's instance.

    Args:
        obj (object): Pointer to agent's class instance.
        state: State to store in object's states list.
        reward: Reward to store in object's rewards list.
        probs: Action's probability distribution to select
            an action and store it in object's action list.
    """

    obj.states.append(state)

    # define action number from a probability distribution
    action = choose_action(probs)

    # one-hot vector to store taken action
    action_vec = np.zeros_like(probs)
    action_vec[action] = 1

    obj.actions.append(action_vec)

    if reward is None:
        reward = 0
    obj.rewards.append(reward)

    # increase reward and timestep accumulators
    obj.episode_reward += reward
    obj.episode_t += 1

    return action


def reset_episode(obj):
    """Resets training auxiliary counters and accumulators

    Args:
        obj (object): Pointer to agent's class instance.
    """

    # episode reward accumulator
    obj.episode_reward = 0

    # episode timestep incrementer
    obj.episode_t = 0

    # lists of rewards, states and actions respectively
    obj.rewards, obj.states, obj.actions = [], [], []
