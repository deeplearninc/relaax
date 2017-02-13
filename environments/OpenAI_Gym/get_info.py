from __future__ import print_function
import sys
import gym

env = gym.make(sys.argv[1])

print('\nAction Space:', env.action_space)

print('Observation Space:', env.observation_space)

print('Timestep Limit:', env.spec.timestep_limit, '\n')
