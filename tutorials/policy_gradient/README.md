### RELAAX tutorial based on simple Policy Gradient

Any RELAAX algorithm should be divided into 4 parts:
 - Client: some simulated environment to interact;
 - Agent: worker connected end-to-end to the Client;
 - Parameter Server: aggregates results from all Agents;
 - Bridge: transport specifications between Agent and Parameter Server.

We'll focused on the last three points to implement some simple Policy Gradient algorithm,
which we used for OpenAI Gym's `CartPole-v0` training.

#### 1. Neural Networks

Let's start from defining a simple neural network class.
We've to have two kind of neural networks:
 - base one for parameter server to accumulate and share parameters between agents
 - agent's neural network, which a bit extends the previous one

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
    def __init__(self, config):
    ...
    # size of the hidden layer for simple FC-NN
    self.layer_size = config.get('hidden_layer_size', 200)
    ...
```

We overwrite the default values of class members by appropriate `yaml` fields if they presence.

And, finally, we define `placeholders` to apply gradients by our `Adam` optimizer:
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

We can get the global weights from parameter server by calling this method of our class at any time:
```python
class GlobalPolicyNN(object):
    ...
    def get_vars(self):
        return self.values
```

**Agent Neural Network**

AgentNN

We define `placeholders` for our weights to assign new values with appropriate method:
```python
class GlobalPolicyNN(object):
    def __init__(self, config):
        ...
        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
            self._assign_values = tf.group(*[
                tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
                ])
        ...

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })
    ...
```

#### 2. Agent

#### 3. Bridge

#### 4. Parameter Server

#### 5. How to Run
