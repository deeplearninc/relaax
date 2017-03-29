from library.core import initialize
from .pgk_network import agent_model


class PGAgent(object):
    def __init__(self, parameter_server):
        self.cfg = PGConfig.preprocess()
        self.ps = parameter_server

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        self.exploit = exploit
        # count global steps between all agents
        self.global_t = 0
        # reset variables used
        # to run single episode
        self.reset_episode()
        # Build TF graph
        self.model = agent_model()
        # Initialize TF
        self.sess = initialize()

        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):

        # beginning of episode
        if self.episode_t == 0:
            self.begin_episode()

        # every episode step
        # will increase episode_t
        # by one if state is not None
        action = self.episode_step(reward, state)

        # end_episode will set episode_t to 0
        if (self.episode_t == self.cfg.batch_size) or terminal:
            self.end_episode()

        return action

    # environment is asking to reset agent
    def reset(self):
        self.reset_episode()
        return True

# Episode states
    # load shared parameters from PS
    def begin_episode(self):
        self.download_ps_weights()

    # every episode step calculate action
    # by running policy and accumulate experience
    def episode_step(self, reward, state):
        # if state is None then skipping this episode
        # there is no action for None state
        if state is None:   # terminal is reached
            self.rewards.append(reward)
            self.episode_t += 1
            return None

        state = state.flatten()
        probs = self.model.predict(state)

        # accumulate experience & retrieve action as simple number
        action = self.act(state, reward, probs)

        return action

    # train policy on accumulated experience
    # and update shared NN parameters
    def end_episode(self):
        if (self.episode_t > 1) and (not self.exploit):
            self.update_ps_weights(self.train_policy())
        self.reset_episode()

    # reset training auxiliary counters and accumulators
    # (also needs to create auxiliary members -> don't move)
    def reset_episode(self):
        self.episode_reward, self.episode_t = 0, 0
        self.rewards, self.states, self.actions = [], [], []

    def choose_action(self, probabilities):
        values = np.cumsum(probabilities)
        r = np.random.rand() * values[-1]
        return np.searchsorted(values, r)


    def act(self, state, reward, probs):
        """Accumulate experience wrt state, actions and reward for agent's instance.

        Args:
            agent (object): Pointer to agent's class instance.
            state: State to store in object's states list.
            reward: Reward to store in object's rewards list.
            probs: Action's probability distribution to select
                an action and store it in object's action list.
        """
        self.states.append(state)

        # define action number from a probability distribution
        action = self.choose_action(probs)

        # one-hot vector to store taken action
        action_vec = np.zeros_like(probs)
        action_vec[action] = 1

        self.actions.append(action_vec)

        if reward is not None:
            self.rewards.append(reward)

        # increase reward and timestep accumulators
        self.episode_reward += reward
        self.episode_t += 1

        return action

    def download_ps_weights(self):
        self.model.set_weights(self.ps.model.get_weights())

    # update PS with agent's gradients
    def update_ps_weights(self, gradients):
        self.ps.sess.run(self.ps.model.apply_gradients, feed_dict={
            p: v for p, v in zip(self.ps.model.gradients, gradients)
        })


    # train policy with accumulated states, rewards and actions
    def train_policy(self):
        return self.sess.run(self.model.compute_gradients, feed_dict={
            self.model.input: self.states,
            self.model.loss.act: self.actions,
            self.model.loss.adv: discounted_reward(np.vstack(self.rewards))
        })
