import tensorflow as tf
import numpy as np
import random
import gym
import math
# import matplotlib.pyplot as plt


def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions),reduction_indices=[1])
        log_like = tf.log(good_probabilities)
        eligibility = log_like * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer, log_like

def run_episode(env, policy_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer, log_like = policy_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []


    for _ in xrange(200):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)

        # advantage: how much better was this action than normal
        advantages.append(future_reward)

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})
    ll = sess.run(log_like, feed_dict={pl_state: states, pl_actions: actions})
    #print 'LL', ll

    return totalreward


env = gym.make('CartPole-v0')
#env.monitor.start('cartpole-hill/', force=True)
policy_grad = policy_gradient()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in xrange(2000):
    reward = run_episode(env, policy_grad, sess)
    print i, reward
t = 0
for _ in xrange(1000):
    reward = run_episode(env, policy_grad, sess)
    t += reward
print t / 1000
