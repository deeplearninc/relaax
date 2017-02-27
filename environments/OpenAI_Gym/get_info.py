from __future__ import print_function
import sys
import gym
from gym.wrappers.frame_skipping import SkipWrapper

skip_4 = SkipWrapper(4)
env = gym.make(sys.argv[1])
env = skip_4(env)

print('\nAction Space:', env.action_space)

print('Observation Space:', env.observation_space)

print('Timestep Limit:', env.spec.timestep_limit, '\n')
