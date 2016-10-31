# DDPG implementation for test
from __future__ import print_function

from ddpg import *
from noise import OUNoise

import gym
from gym.spaces import Box


def trainer():
    env = gym.make(experiment)
    steps = env.spec.timestep_limit                 # Steps per Episode via gym API call
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"

    # Randomly initialize critic, actor, target critic, target actor network and replay buffer
    agent = DDPG(env, is_batch_norm)
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter = 0
    reward_per_episode = 0
    total_reward = 0
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    print("Number of States:", num_states)
    print("Number of Actions:", num_actions)
    print("Number of Steps per episode:", steps)
    # saving reward:
    reward_st = np.array([0])

if __name__ == '__main__':
    trainer()
