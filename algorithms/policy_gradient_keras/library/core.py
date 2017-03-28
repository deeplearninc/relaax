from .config import *   # PGConfig
from .loss import *
from .reward import *


def compute_gradients(agent):
    return tf.gradients(agent.loss.eval, agent.net.trainable_weights)


def apply_gradients(agent, optimizer_name):
    optimizer = None
    agent.gradients =\
        [tf.placeholder(v.dtype, v.get_shape()) for v in agent.net.trainable_weights]

    if optimizer_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=agent.cfg.learning_rate)

    if optimizer is not None:
        return optimizer.apply_gradients(zip(agent.gradients, agent.net.trainable_weights))


def initialize():
    session = tf.Session()
    keras.backend.set_session(session)
    session.run(tf.variables_initializer(tf.global_variables()))
    return session


# === Session dependent functions below ===

# update PS with agent's gradients
def update_global_weights(agent, gradients):
    feed_dict = {p: v for p, v in zip(agent.ps.model.net.gradients, gradients)}
    agent.sess.run(agent.ps.model.apply_gradients, feed_dict=feed_dict)


# train policy with accumulated states, rewards and actions
def train_policy(obj):
    return obj.sess.run(obj.agent.compute_gradients, feed_dict={
        obj.agent.net.input: obj.states,
        obj.agent.loss.act: obj.actions,
        obj.agent.loss.adv: discounted_reward(np.vstack(obj.rewards))})


# === Move below to some appropriate library parts ===

def choose_action(probabilities):
    values = np.cumsum(probabilities)
    r = np.random.rand() * values[-1]
    return np.searchsorted(values, r)


def accumulate(agent, state, reward, probs):
    """Accumulate experience wrt state, actions and reward for agent's instance.

    Args:
        agent (object): Pointer to agent's class instance.
        state: State to store in object's states list.
        reward: Reward to store in object's rewards list.
        probs: Action's probability distribution to select
            an action and store it in object's action list.
    """
    agent.states.append(state)

    # define action number from a probability distribution
    action = choose_action(probs)

    # one-hot vector to store taken action
    action_vec = np.zeros_like(probs)
    action_vec[action] = 1

    agent.actions.append(action_vec)

    if reward is not None:
        agent.rewards.append(reward)

    # increase reward and timestep accumulators
    agent.episode_reward += reward
    agent.episode_t += 1

    return action