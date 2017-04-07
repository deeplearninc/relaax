import numpy as np

rew_list = [1, 0, 0, 1, 0,
            0, 0, 1, 1, 1]

advantages = []
for index in range(len(rew_list)):
    future_reward = 0
    future_transitions = len(rew_list) - index
    decrease = 1
    for index2 in xrange(future_transitions):
        future_reward += rew_list[index2 + index] * decrease
        decrease = decrease * 0.97
    advantages.append(future_reward)

cloumn_adv = np.expand_dims(advantages, axis=1)
print(cloumn_adv)


def discounted_reward(rewards, gamma):
    # take 1D float array of rewards and compute discounted reward
    rewards = np.vstack(rewards)
    discounted_reward = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_reward[t] = running_add
    # size the rewards to be unit normal
    # it helps control the gradient estimator variance
    #discounted_reward -= np.mean(discounted_reward)
    #discounted_reward /= np.std(discounted_reward) + 1e-20

    return discounted_reward

print(discounted_reward(rew_list, gamma=0.97))
