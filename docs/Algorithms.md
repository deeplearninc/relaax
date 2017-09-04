## [Algorithms](../README.md#contents)
> click on title to go to contents
- [Distributed A3C](#distributed-a3c)
    - [Distributed A3C Architecture](#distributed-a3c-architecture)
    - [Intrinsic Curiosity Model for DA3C](#intrinsic-curiosity-model-for-da3c)
    - [Distributed A3C Config](#distributed-a3c-config)
    - [Performance on some of the Atari Environments](#performance-on-some-of-the-atari-environments)
- [Distributed A3C Continuous](#distributed-a3c-continuous)
    - [Distributed A3C Architecture with Continuous Actions](#distributed-a3c-architecture-with-continuous-actions)
    - [Performance on gym's Walker](#performance-on-gyms-walker)
- [Distributed TRPO with GAE](#distributed-trpo-with-gae)
    - [Performance on gym's BipedalWalker](#performance-on-gyms-bipedalwalker)
- [Other Algorithms](#other-algorithms)

### [Distributed A3C](#algorithms)
Inspired by original [paper](https://arxiv.org/abs/1602.01783) - 
Asynchronous Methods for Deep Reinforcement Learning from [DeepMind](https://deepmind.com/)

It is actor-critic algorithm, which learns both a policy and a state-value function, and the value function
is used for bootstrapping, i.e., updating a state from subsequent estimates, to reduce variance and
accelerate learning.

In DA3C, parallel actors employ different exploration policies to stabilize training, so that experience
replay is not utilized. Different from most deep learning algorithms, this distributed method can
run on the multiple nodes with centralised parameter server. For Atari games, DA3C ran much faster yet
performed better than or comparably with DQN, Gorila, D-DQN, Dueling D-DQN, and Prioritized D-DQN.
Furthermore DA3C also performs better than original A3C or it synchronous variant, which called A2C. 
DA3C also succeeded on continuous motor control problems: TORCS car racing games and MujoCo physics
manipulation and locomotion, and Labyrinth, a navigating task in random 3D mazes using visual inputs,
in which an agent will face a new maze in each new episode, so that it needs to learn a general strategy
to explore random mazes. 

There are original pseudo code for A3C:

![img](resources/A3C-pseudo_code.png)

DA3C maintains a policy ![img](http://latex.codecogs.com/svg.latex?%5Cpi%5Cleft%28a_%7Bt%7D%5Cmid%5C%5Cs_%7Bt%7D%3B%5C%2C%5Ctheta%5Cright%29)
and an estimate of the value function ![img](http://latex.codecogs.com/svg.latex?V%5Cleft%28s_%7Bt%7D%3B%5Ctheta_%7B%5Cupsilon%7D%5Cright%29),
being updated with _n_-step returns in the forward view, after every _t_<sub>_max_</sub> actions or
reaching a terminal state, similar to using minibatches. In contrast to the original code we use processes
for the agents instead of threads, where each agent or some set of agents can be run on separate node.   

The gradient update can be represented with _TD_-error multiplier as in original paper
![img](http://latex.codecogs.com/svg.latex?%5Cbigtriangledown_%7B%7B%5Ctheta%7D%27%7Dlog%5C%2C%5Cpi%5Cleft%28a_%7Bt%7D%5Cmid%5C%5Cs_%7Bt%7D%3B%5C%2C%7B%5Ctheta%7D%27%5Cright%29%5Cleft%28R-V%5Cleft%28s_%7Bt%7D%3B%7B%5Ctheta%7D%27_%7B%5Cupsilon%7D%5Cright%29%5Cright%29)
or with an estimate of the _advantage_ function:
![img](http://latex.codecogs.com/svg.latex?%5Cbigtriangledown_%7B%7B%5Ctheta%7D%27%7Dlog%5C%2C%5Cpi%5Cleft%28a_%7Bt%7D%5Cmid%5C%5Cs_%7Bt%7D%3B%5C%2C%7B%5Ctheta%7D%27%5Cright%29A%5Cleft%28s_%7Bt%7D%2Ca_%7Bt%7D%3B%5C%2C%5Ctheta%2C%5Ctheta_%7B%5Cupsilon%7D%5Cright%29),
where
![img](http://latex.codecogs.com/svg.latex?A%5Cleft%28s_%7Bt%7D%2Ca_%7Bt%7D%3B%5C%2C%5Ctheta%2C%5Ctheta_%7B%5Cupsilon%7D%5Cright%29%3D%5Csum_%7Bi%3D0%7D%5E%7Bk-1%7D%5Cgamma%5E%7Bi%7Dr_%7Bt%2Bi%7D%2B%5Cgamma%5E%7Bk%7DV%5Cleft%28s_%7Bt%2Bk%7D%3B%5Ctheta_%7B%5Cupsilon%7D%5Cright%29-V%5Cleft%28s_%7Bt%7D%3B%5Ctheta_%7B%5Cupsilon%7D%5Cright%29)
with _k_ upbounded by _t_<sub>_max_</sub>.
To use the last one just set config parameter `use_gae` to `true`.
The full set of possible config setups would be described later.
Gradients are also applied with delay compensation on the server parameter for better convergence.  

#### [Distributed A3C Architecture](#algorithms)
![img](resources/DA3C-Architecture.png)

**Environment (Client)** - each client connects to a particular Agent (Learner).

The main role of any client is feeding data to an Agent by transferring:
state, reward and terminal signals (for episodic tasks if episode ends).
Client updates these signals at each time step by receiving the action
signal from an Agent and then sends updated values back.

- _Process State_: each state could be pass through some filtering
procedure before transferring (if you defined). It could be some color,
edge or blob transformations (for image input) or more complex
pyramidal, Kalman's and spline filters.

**Agent (Parallel Learner)** - each Agent connects to the Parameter Server.

The main role of any agent is to perform a main training loop.
Agent synchronize their neural network weights with the global network
by copying the last one weights at the beginning of each training mini loop.
Agent executes N steps of Client's signals receiving and sending actions back.
These N steps is similar to batch collection. If batch is collected
Agent computes the loss (wrt collected data) and pass it to the Optimizer.
It could be some SGD optimizer (ADAM or RMSProp) which computes gradients
and sends it to the Parameter Server for update of its neural network weights.
All Agents works absolutely independent in asynchronous way and can update or 
receive the global network weights at any time.

- _Agent's Neural Network_: we use the neural network architecture similar to [universe agent](https://github.com/openai/universe-starter-agent/blob/master/model.py) (by default).
    - _Input_: `3D` input to pass through `2D` convolutions (default: `42x42x1`).
    - _Convolution Layers : `4` layers with `32` filters each and `3x3` kernel, stride `2`, `ELU` activation (by default).
    - _Fully connected Layers_: one layer with `256` hidden units and `ReLU` activation (by default).
    - _LSTM Layers_: one layer with `256` cell size (by default it's replaced with fully connected layer).
    - _Actor_: fully connected layer with number of units equals to `action_size` and `Softmax` activation (by default).  
    It outputs an `1-D` array of probability distribution over all possibly actions for the given state.
    - _Critic_: fully connected layer with `1` unit (by default).  
    It outputs an `0-D` array representing the value of an state (expected return from this point).

- _Total Loss_ _`= Policy_Loss + critic_scale * Value_Loss`_  
    It uses `critic_scale` parameter to set a `critic learning rate` relative to `policy learning rate`  
    It's set to `0.5` by default, i.e. the `critic learning rate` is `2` times smaller than `policy learning rate`
    
    - _Value Loss_: sum (over all batch samples) of squared difference between
    expected discounted reward `(R)` and a value of the current sample state - `V(s)`,
    i.e. expected discounted return from this state.  
    ![img](http://latex.codecogs.com/svg.latex?0.5%2A%5Csum_%7Bt%3D0%7D%5E%7Bt_%7Bmax%7D-1%7D%5Cleft%28R_%7Bt%7D-V%5Cleft%28s_%7Bt%7D%3B%5Ctheta_%7B%5Cupsilon%7D%5Cright%29%5Cright%29%5E%7B2%7D),
    where ![img](http://latex.codecogs.com/svg.latex?R_%7Bt%7D%3D%5Csum_%7Bi%3D0%7D%5E%7Bk-1%7D%5Cgamma%5E%7Bi%7Dr_%7Bt%2Bi%7D%2B%5Cgamma%5E%7Bk%7DV%5Cleft%28s_%7Bt%7D%3B%5Ctheta_%7B%5Cupsilon%7D%5Cright%29)
    with _k_ upbounded by _t_<sub>_max_</sub>.  
    If _s<sub>t</sub>_ is terminal then _V(s<sub>t</sub>) = 0_.

    - _Policy Loss_:  
    ![img](http://latex.codecogs.com/svg.latex?-%5Csum_%7Bt%3D1%7D%5E%7Bt_%7Bmax%7D%7D%5Cleft%28%5C%2Clog%5C%2C%5Cpi%28a_%7Bt%7D%5Cmid%5C%5Cs_%7Bt%7D%3B%7B%5Ctheta%7D%27%29%5C%2CA%28s_%7Bt%7D%2Ca_%7Bt%7D%3B%5C%2C%5Ctheta%2C%5Ctheta_%7B%5Cupsilon%7D%29-%5Cbeta%5C%2C%5Cpi%28%5Ccdot%5Cmid%5C%5Cs_%7Bt%7D%3B%7B%5Ctheta%7D%27%29%5Clog%5C%2C%5Cpi%28%5Ccdot%5Cmid%5C%5Cs_%7Bt%7D%3B%7B%5Ctheta%7D%27%29%5C%2C%5Cright%29)  
     where the 1-st term is multiplication of policy log-likelihood on advantage function,  
     and the last term is entropy multiplied by regularization parameter `entropy_beta = 0.01` (by default).

- _Compute Gradients_: it computes the gradients wrt neural network weights and total loss.   
    Gradients are also clipped wrt parameter `gradients_norm_clipping = 40.0` (by default).  
    To perform the clipping, the values `nn_weights[i]` are set to:

      nn_weights[i] * clip_norm / max(global_norm, clip_norm)

    where:

      global_norm = sqrt(sum([l2norm(w)**2 for w in nn_weights]))

    If `clip_norm > global_norm` then the entries in `nn_weights` remain as they are,  
    otherwise they're all shrunk by the global ratio.  
    To avoid clipping just set `gradients_norm_clipping = false` in config yaml.

- _Synchronize Weights_: it synchronize agent's weights with global neural network by copying  
    the last own to replace its own at the beginning of each batch collection step
    (_1..t_<sub>_max_</sub> or _terminal_).  
    The new step will not start until the weights are updated, but it allows to switch  
    this procedure in non-blocking manner by setting `hogwild` to `true` in the code.

- _Softmax Action_: it uses a `Boltzmann` distribution to select an action, so it chooses more  
    often actions, which has more probability and we called this `Softmax` action for simplicity.  
    This method has some benefits over classical _e_-greedy strategy and helps to avoid problem  
    of "path along the cliff". Furthermore it helps to explore more at the beginning of the training.  
    Agent becomes more confident in some actions while training and the probability distribution  
    over actions is becoming more acute.

**Parameter Server (Global)** - one for whole algorithm (training process).

The main role of the Parameter Server is to synchronize neural networks weights between Agents.  
It holds the shared (global) neural network weights, which is updated by the Agents gradients,  
and sent the actual copy of its weights back to Agents to synchronize.

- _Global Neural Network_: neural network weights is similar to Agent's one.

- _Some SGD Optimizer_: it holds a SGD optimizer and its state (`Adam | RMSProp`).  
    It is one for all Agents and used to apply gradients from them.  
    The default optimizer is `Adam` with `initial_learning_rate = 1e-4`  
    since the last one is linear annealing wrt `max_global_step` parameter.

#### [Intrinsic Curiosity Model for DA3C](#algorithms)

`DA3C` algorithm can also be extended with additional models.  
By default it can use a [ICM](https://arxiv.org/abs/1705.05363) by setting `use_gae` parameter to `True`.

`ICM` helps Agent to discover an environment out of curiosity when extrinsic rewards are spare
or not present at all. This model proposed an intrinsic reward which is learned jointly with Agent's policy
even without any extrinsic rewards from the environment. Conceptual architecture is shown in figure below:

![img](resources/ICM-Architecture.png)

#### [Distributed A3C Config](#algorithms)

You must specify the parameters for the algorithm in the corresponding `app.yaml` file to run:

    algorithm:
        name: da3c                  # name of the algorithm to load

    input:
        shape: [42, 42]             # shape of the incoming state from an environment
        history: 4                  # number of consecutive states to stack for input
        use_convolutions: true      # set to True to use convolution layers after input

    output:
        continuous: false           # set to True to use continuous Actor
        action_size: 18             # action size for the given environment

    batch_size: 5                   # t_max for batch collection step size
    hidden_sizes: [256]             # list to define layers sizes after convolutions

    use_icm: true                   # set to True to use ICM module
    use_gae: true                   # set to True to use generalized advantage estimation
    gae_lambda: 1.00                # discount lambda for generalized advantage estimation

    use_lstm: true                  # set to True to use LSTM instead of Fully-Connected layers
    max_global_step: 1e8            # amount of maximum global steps to pass through the training

    optimizer: Adam
    initial_learning_rate: 1e-4     # initial learning rate which linear annealing through training

    entropy_beta: 0.01              # entropy regularization constant
    rewards_gamma: 0.99             # rewards discount factor
    gradients_norm_clipping: 40.    # value for gradients norm clipping

    icm:                            # ICM relevant parameters
        nu: 0.01                    # prediction bonus multiplier for intrinsic reward
        beta: 0.2                   # forward loss importance against inverse model
        lr: 1e-3                    # ICM learning rate

We use some notations to outline different versions of the `DA3C`.  
So, 

**DA3C Graph sample from Tensorboard**

![img](resources/DA3C-Graph.png)

#### [Performance on some of the Atari Environments](#algorithms)
Breakout with DA3C-FF and 8 parallel agents: score performance is similar to DeepMind [paper](https://arxiv.org/pdf/1602.01783v2.pdf#19)
![img](resources/Breakout-8th-80mil.png "Breakout")

Boxing with DA3C-FF and 8 parallel agents: ih this case we outperforms significantly DeepMind, but
we have some instability in training process (anyway DeepMind shows only 34 points after 80mil steps)
![img](resources/Boxing-8th-35mil.png "Boxing")

### [Distributed A3C Continuous](#algorithms)
Distributed version of A3C algorithm, which can cope with continuous action space.
Inspired by original [paper](https://arxiv.org/abs/1602.01783) - 
Asynchronous Methods for Deep Reinforcement Learning from [DeepMind](https://deepmind.com/)

#### [Distributed A3C Architecture with Continuous Actions](#algorithms)
![img](resources/DA3C-Continuous.png)

Most of the parts are the same to previous scheme, excluding:

- _Signal Filtering_: perform by Zfilter `y = (x-mean)/std` using running estimates of mean and std
 inspired by this [source](http://www.johndcook.com/blog/standard_deviation/). You can filter both
 states and rewards. We use it only for states by default.

- _Agent's (Global) Neural Network_: we use the similar architecture to [A3C paper](https://arxiv.org/pdf/1602.01783v2.pdf#12).
    Each continuous state passes some filtering procedure before transferring to _Input_ by default.
    - _Input_: vector of filtered state input (default: 24).
    - _Fully connected Layer_: consists of 128 hidden units, then ReLU applies (by default).
    - _LSTM_: consists of 128 memory cells (by default).
    - _Value_: outputs one value without applying of additional operators (by default).
    It is Critic's output, which represents value function output V(s) - how well this state (equals to expected return from this point).
    - _Policy_: Actor's output is divided separately on `mu` and `sigma`
        - _mu_: scalar of linear output.
        - _sigma_: applying SoftPlus operator, outputs a scalar.

    You can also specify your own architecture in provided JSON file.

- _Choose Action_: we use a random sampling wrt given `mu` and `sigma`

- _Total Loss_: it's scalar sum of value and policy loss.
    - _Value Loss_: the same to previous scheme.
    - _Policy Loss_: `GausNLL * TD + entropy`

    `GausNLL` is gaussian negative-log-likelihood

    `GausNLL = (sum(log(sigma), index=1) + batch_size * log(2*pi))/2 - sum(power, index=1)`,

     where `power = (A - mu)^2 * exp(-log(sigma)) * -0.5` - produce column vector.

     `TD = (R - V)` - temporary difference between total discounted reward (R)
     and a value of the current sample state V(s).

    `entropy = -sum(0.5 * log(2 * pi * sigma) + 1, index=1) * entropy_beta_coefficient`,

     resulting sparse matrix we sum over rows to produce column vector.

    `entropy_beta_coefficient = 0.001`

We also use a smaller `learning rate = 1e-4`

#### [Performance on gym's Walker](#algorithms)
![img](resources/a3c_cont-4th-80mil.png "Walker")

##### [Server Latency](#algorithms)
Measure how fast Agent returns Action in response to the State sent by the Client

| Node Type  | Number of clients | Latency  |
| ---------- |:-----------------:|:--------:|
| c4.large   |           4       | ???ms    |
| c4.large   |           8       | ???ms    |
| c4.large   |          16       | ???ms    |
| m4.xlarge  |          32       | 323ms    |
| m4.xlarge  |          48       | ???ms    |
| m4.xlarge  |          64       | ???ms    |
| c4.xlarge  |          32       | ???ms    |
| c4.xlarge  |          48       | ???ms    |
| c4.xlarge  |          64       | ???ms    |
| c4.xlarge  |          96       | ???ms    |
| c4.xlarge  |          128      | ???ms    |
| c4.2xlarge |          232      | ???ms    |
| c4.2xlarge |          271      | ???ms    |

TBD - Latency chart (Show latency of the agents over time)


##### [Compute Performance with different amount of clients and node types (AWS)](#contents)

| Node Type  | Number of clients | Performance       |
| ---------- |:-----------------:| -----------------:|
| m4.xlarge  |          32       | 99 steps per sec  |
| m4.xlarge  |          48       | 167 steps per sec |
| m4.xlarge  |          64       | 171 steps per sec |
| c4.xlarge  |          48       | 169 steps per sec |
| c4.xlarge  |          64       | 207 steps per sec |
| c4.xlarge-m4.xlarge | 64       | 170 steps per sec |
| c4.xlarge-m4.xlarge | 96       | 167 steps per sec |
| c4.xlarge-m4.xlarge | 128      | 177 steps per sec |
| c4.2xlarge |          232      | 232 steps per sec |
| c4.2xlarge |          271      | 271 steps per sec |
<br><br>


### [Distributed TRPO with GAE](#algorithms)
Distributed version of TRPO-GAE algorithm, which can cope with both continuous & discrete action space.

Inspired by original papers:

- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

The main pipeline of the algorithm is the similar to the original sources, but collecting of
trajectories is performed independently by parallel agents. These agents have a copy of
policy neural network to rollout trajectories from its client.
Parameter server is blocked to update when the batch is collected and this procedure repeats.

#### [Performance on gym's BipedalWalker](#algorithms)
`batch_size == 10.000, trajectory_length == 1600, parallel_agents == 8`
![img](resources/bipedal-walker-trpo-10k-control.png "BipedalWalker")
<br><br>

### [Other Algorithms](#algorithms)
These other algorithms we are working on and planning to make them run on RELAAX server:

* ACER (A3C with experience)
Inspired by:
    - [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)

* UNREAL
Inspired by:
    - [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)

* Distributed DQN (Gorila)
Inspired by:
    - [Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1507.04296)

* PPO with L-BFGS (similar to TRPO)
Inspired by:
    - [John Schulman's modular_rl repo](https://github.com/joschu/modular_rl)

* CEM
Inspired by:
    - [Cross-Entropy Method for Reinforcement Learning](https://esc.fnwi.uva.nl/thesis/centraal/files/f2110275396.pdf)

* DDPG
Inspired by:
    - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
