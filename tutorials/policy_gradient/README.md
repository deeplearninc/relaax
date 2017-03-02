### RELAAX tutorial based on simple Policy Gradient

Any RELAAX algorithm should be divided into 4 parts:
 - Client: some simulated environment to interact;
 - Agent: worker, connected end-to-end to the Client;
 - Parameter Server: aggregates results from all Agents;
 - Bridge: transport specifications between Agent and Parameter Server.

We'll focused on the last three points to implement some simple Policy Gradient algorithm,
which we used to train OpenAI Gym's `CartPole-v0`
<br></br>

#### 1. Neural Networks

Let's start from defining a simple neural network class.
We've to have two kind of neural networks:
 - base one for parameter server to accumulate and share parameters between agents
 - agent's neural network, which extends a bit the previous one
<br></br>

**Parameter Server Neural Network**

We can define the beginning of neural network class as follows:
```python
class GlobalPolicyNN(object):
    # This class is used for global-NN and holds only weights on which applies computed gradients
    def __init__(self, config):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)
    ...
```

As you can see, we set the `tensorflow` variable for the global step counter,
which accumulates all steps passed by each parallel agent (worker).
We also define an appropriate `tensorflow` operation to increment these variable.

Then we want to define the weights for our neural network.
Our neural network has one hidden layer for simplicity (without bias).
We need to define 2 weight matrices:
 - from input to hidden
 - form hidden to output

```python
    def __init__(self, config):
    ...
        self.W1 = tf.get_variable('W1', shape=[config.state_size, config.layer_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable('W2', shape=[config.layer_size, self._action_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.values = [
            self.W1, self.W2
        ]
    ...
```

We use `xavier` initialization type for our weights.
You can ask a reasonable question: what is `config`?
We use a specific config file for each experiment to setup parameters in some flexible way.
You can find one there (in this folder) named `policy_gradient.yaml`:

```yaml
  ...
  action_size: 1                  # action size for the given environment (CartPole-v0)
  state_size:  4                  # size of the input observation (flattened)
  hidden_layer_size: 10           # size of the hidden layer for simple FC-NN
  ...
```

This `yaml` is read at initialize procedure and stored in `Config class` field by field with `config.py`

```python
class Config(relaax.algorithm_base.config_base.ConfigBase):
    def __init__(self, config):
        ...
        # size of the hidden layer for simple FC-NN
        self.layer_size = config.get('hidden_layer_size', 200)
        ...
```

We overwrite the default values of class members by appropriate `yaml` fields if they presence.

And, finally, we define `placeholders` to apply gradients by `Adam` optimizer:
```python
class GlobalPolicyNN(object):
    def __init__(self, config):
        ...
        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self.learning_rate = config.learning_rate

    ...
    def apply_gradients(self):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=self.learning_rate
    )
    self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, self.values))
    return self
```
You can free to use any other optimizer, but you have to define related parameters in `yaml` and `config.py`

We can get the global weights from our parameter server by calling this method at any time:
```python
class GlobalPolicyNN(object):
    ...
    def get_vars(self):
        return self.values
```
<br></br>

**Agent Neural Network**

We use Agent's network to rollout the client environment.
Since that we have to define connections for forward pass through our network:
```python
class AgentPolicyNN(GlobalPolicyNN):
    # This class additionally implements loss computation and gradients wrt this loss
    def __init__(self, config):
        super(AgentPolicyNN, self).__init__(config)

        # state (input)
        self.s = tf.placeholder(tf.float32, [None, config.state_size])

        hidden_fc = tf.nn.relu(tf.matmul(self.s, self.W1))

        # policy (output)
        self.pi = tf.nn.sigmoid(tf.matmul(hidden_fc, self.W2))
        ...

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]
    ...
```

It needs to define `placeholder` for the input state
with first unset dimension to have flexibility in batch size.
Standard ReLU (Rectifier Liner Unit) function is applied to the output of the hidden layer.
And we use `sigmoid` at the final output to represent probability of action to take.

Forward pass trough the network is performed by `run_policy` method,
which takes a single state as input.

Then we define `placeholders` for network weights to assign them a new values with appropriate method:
```python
class GlobalPolicyNN(object):
    def __init__(self, config):
        ...
        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
            self._assign_values = tf.group(*[
                tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
                ])

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })
    ...
```

To compute the `gradients` wrt our weights we have to define a `loss` function:
```python
class GlobalPolicyNN(object):
    ...
    def compute_gradients(self):
        self.grads = tf.gradients(self.loss, self.values)
        return self

    def prepare_loss(self):
        self.a = tf.placeholder(tf.float32, [None, self._action_size], name="taken_action")
        self.advantage = tf.placeholder(tf.float32, name="discounted_reward")

        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.
        log_like = tf.log(self.a * (self.a - self.pi) + (1 - self.a) * (self.pi - self.a))
        self.loss = -tf.reduce_mean(log_like * self.advantage)

        return self
```
<br></br>

#### 2. Agent

Agent interacts with its client (simulated environment) by
receiving states, rewards and terminal as input signals, and
sends the actions, defined by its neural network forward pass.
It's a typical reinforcement learning procedure.

In addition wrt distributed training scenario:
we have a set of parallel agents, that collecting
experience from its own clients and send it to
parameter server as computed gradients wrt loss
of this experience.

Parameter server performs updates and agents are synchronized
with it by updating its own neural network weights
before starting of each experience collecting loop.

To implement the Agent, besides issues with parameter server,
we have to define 3 public methods from agent's interface:
 - `act`
 - `reward_and_act`
 - `reward_and_reset`

Constructor of the Agent's class looks like a follows:
```python
class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server
        self._local_network = make_network(config)

        self.global_t = 0           # counter for global steps between all agents
        self.episode_reward = 0     # score accumulator for current episode (game)

        self.states = []            # auxiliary states accumulator through batch_size = 0..N
        self.actions = []           # auxiliary actions accumulator through batch_size = 0..N
        self.rewards = []           # auxiliary rewards accumulator through batch_size = 0..N

        initialize_all_variables = tf.variables_initializer(tf.global_variables())

        self._session = tf.Session()

        self._session.run(initialize_all_variables)

        # copy weights from parameter server (shared) to local agent
        self._local_network.assign_values(self._session, self._parameter_server.get_values())
    ...
```

We've got the `parameter_server` as our member for further interaction.

By calling `make_network(config)` we define an Agent's neural network, which we described above.

We also need some auxiliary accumulators within the experience collecting loop
also as a `global_step` counter and episode score to print out it through the training process.

Let's define 3 methods for interaction with the client:
```python
class Agent(relaax.algorithm_base.agent_base.AgentBase):
    ...
    def act(self, state):
        start = time.time()

        # Run the policy network and get an action to take
        prob = self._local_network.run_policy(self._session, state)
        action = 0 if np.random.uniform() < prob else 1

        self.states.append(state)
        self.actions.append([action])

        self.metrics().scalar('server latency', time.time() - start)

        return action

    def reward_and_act(self, reward, state):
        if self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if not self._reward(reward):
            return None

        print("Score =", self.episode_reward)
        score = self.episode_reward

        self.metrics().scalar('episode reward', self.episode_reward)
        self.episode_reward = 0

        self._update_global()

        self.states, self.actions, self.rewards = [], [], []
        return score

    def _reward(self, reward):
        self.episode_reward += reward
        self.rewards.append(reward)

        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step
    ...
```

The `act` method is directly called from the client for the first
interaction only. When client initialized its own state it has the only
starting state and no rewards or terminal signals. It needs to just
call `act` with this state for the first time. If client environment
reaches an terminal state (if it exists) it needs to repeat this call
after reinitialization from the terminal state.

For the next interactions `act` method is called through `reward_and_act`
or, if we reached the terminal state, `reward_and_reset` method is taken.

We get an `action` with some probability (unity uniform):
```python
    action = 0 if np.random.uniform() < prob else 1
```

We try to skew our logits (`prob`) to `1` for the `0` action
and to `0` for `1` action through the training.

And we add this `action` as a list to represent as `one-hot` vector:
```python
    self.actions.append([action])
```

We perform a gradients computation at each terminal signal by the
`self._update_global()` which looks like as follows:
(our collecting loop of experience is equal to `1` episode at this case)
```python
class Agent(relaax.algorithm_base.agent_base.AgentBase):
    ...
    def _update_global(self):
        feed_dict = {
            self._local_network.s: self.states,
            self._local_network.a: self.actions,
            self._local_network.advantage: self.discounted_reward(np.vstack(self.rewards)),
        }

        self._parameter_server.apply_gradients(
            self._session.run(self._local_network.grads, feed_dict=feed_dict)
        )

        # copy weights from shared to local
        self._local_network.assign_values(self._session, self._parameter_server.get_values())
    ...
```

Computed gradients immediately send to `parameter_server` via `Bridge` (see below) to apply.
Then we synchronized agent's weights from `parameter_server` to be sure
that we grub all past updates (also from other agents) before new batch collecting.

We also defined function for discounted reward as follows:
```python
class Agent(relaax.algorithm_base.agent_base.AgentBase):
    ...
    def discounted_reward(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self._config.GAMMA + r[t]
            discounted_r[t] = running_add
        # size the rewards to be unit normal (helps control the gradient estimator variance)
        # discounted_r = discounted_r.astype(np.float64)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r) + 1e-20
        return discounted_r
```

#### 3. Bridge

#### 4. Parameter Server

#### 5. How to Run
