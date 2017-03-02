### RELAAX tutorial based on simple Policy Gradient

Any RELAAX algorithm should be divided into 4 parts:
 - Client: some simulated environment to interact;
 - Agent: worker connected end-to-end to the Client;
 - Parameter Server: aggregates results from all Agents;
 - Bridge: transport specifications between Agent and Parameter Server.

We'll focused on the last three points to implement some simple Policy Gradient algorithm,
which we used for OpenAI Gym's `CartPole-v0` training.
<br></br>

#### 1. Neural Networks

Let's start from defining a simple neural network class.
We've to have two kind of neural networks:
 - base one for parameter server to accumulate and share parameters between agents
 - agent's neural network, which a bit extends the previous one
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
We need to define 2 weight matrix:
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

This `yaml` is read at initialize procedure and stored in `Config class` field by field in `config.py`

```python
class GlobalPolicyNN(object):
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
Since that we have to define connections for forward pass through the network:
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
And we use `sigmoid` at the final output to represent probability of action take.

Forward pass trough the network is performed by `run_policy` method,
which takes a single states as input.

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

#### 3. Bridge

#### 4. Parameter Server

#### 5. How to Run
