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

**Agent Neural Network**

efefe

#### 2. Agent

#### 3. Bridge

#### 4. Parameter Server

#### 5. How to Run
