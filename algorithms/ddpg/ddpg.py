import tensorflow as tf
import numpy as np
from params import *

from replay_buffer import ReplayBuffer
from ou_noise import OUNoise


class DDPG:
    """ Deep Deterministic Policy Gradient Algorithm"""
    def __init__(self, env, is_batch_norm_):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.sess = tf.Session()

        if is_batch_norm_:
            from actor_net_bn import ActorNet
            from critic_net_bn import CriticNet
            self.actor_net = ActorNet(self.sess, self.state_dim, self.action_dim)
            self.critic_net = CriticNet(self.sess, self.state_dim, self.action_dim)
        else:
            from actor_net import ActorNet
            from critic_net import CriticNet
            self.actor_net = ActorNet(self.sess, self.state_dim, self.action_dim)
            self.critic_net = CriticNet(self.sess, self.state_dim, self.action_dim)

        # Update targets
        self.sess.run(tf.initialize_all_variables())
        self.actor_net.update_target()
        self.critic_net.update_target()

        # Initialize Memory Replay Buffer:
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize an Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)

        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        next_action_batch = self.actor_net.target_actions(next_state_batch)
        q_value_batch = self.critic_net.target_q(next_state_batch, next_action_batch)

        # Calculate y_batch
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])

        # Update critic by minimizing the loss L
        self.critic_net.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_net.actions(state_batch)
        q_gradient_batch = self.critic_net.gradients(state_batch, action_batch_for_gradients)

        self.actor_net.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_net.update_target()
        self.critic_net.update_target()

    def action(self, state):
        return self.actor_net.action(state)

    def noise_action(self, state):
        # Select action according to the current policy and exploration noise
        return self.action(state) + self.exploration_noise.noise()

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t, a_t, r_t, s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.train()

        # Reinitialize the random OU process when an episode ends
        if done:
            self.exploration_noise.reset()
