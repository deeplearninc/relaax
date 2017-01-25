import sys
import gym

env = gym.make(sys.argv[1])

print('Action Space:', env.action_space)

print('Observation Space:', env.observation_space)

print('Timestep Limit:', env.spec.timestep_limit)
