# DDPG implementation
from __future__ import print_function

from ddpg import *
import gym
from gym.spaces import Box


def trainer():
    env = gym.make(experiment)
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"

    # Initialize critic, actor, target critic, target actor network and replay buffer
    agent = DDPG(env, is_batch_norm)
    reward_per_episode = 0
    total_reward = 0

    steps = env.spec.timestep_limit  # Steps per Episode via gym API call
    print("Number of States:", env.observation_space.shape[0])
    print("Number of Actions:", env.action_space.shape[0])
    print("Number of Steps per episode:", steps)

    for i in range(episodes):
        # print("==== Starting episode no:", i, "====", "\n")
        state = env.reset()
        reward_per_episode = 0
        for step in range(steps):
            # rendering environment (optional)
            # env.render()

            action = agent.noise_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state

            # Action output (optional)
            # print("Action at step", step, " :", action, "\n")
            reward_per_episode += reward

            # check if episode ends:
            if done or (step == steps - 1):
                print('EPISODE: ', i, ' Steps: ', step, ' Total Reward: ', reward_per_episode)
                break
    total_reward += reward_per_episode
    print("Average reward per episode {}".format(total_reward / episodes))

if __name__ == '__main__':
    trainer()
