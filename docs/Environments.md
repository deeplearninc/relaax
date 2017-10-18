### [Supported Environments](../README.md#contents)
> click on title to go to contents
- [Basic](#basic)
- [OpenAI Gym](#openai-gym)
- [DeepMind Lab](#deepmind-lab)
- [Customized](#customized)

RELAAX allows to **generate** some types of predefined apps by `relaax new` command.  
The set of possible options and usage is displayed by `relaax new --help`:  
```bash
Usage: relaax new [OPTIONS] APP_NAME

  Create new RELAAX application.

Options:
  -a, --algorithm     [policy-gradient|da3c|trpo|ddpg|dqn]
                      Algorithm to use
                      with this application. [default: policy-gradient]
  -e, --environment   [basic|openai-gym|deepmind-lab] 
                      Environment to base
                      application on.  [default: basic]
  --help              Show this message and exit.
``` 

It is necessary to provide some `app.yaml` to each application, where all necessary configs is setup.
There are two important sections, which is responsible for appropriate options. 
```yaml
environment:
  run: executable       # path to script (wrapper), which is launch an environment or directly to some executable 
  max_episodes: 10000   # amount of episodes to pass in through the training
  infinite_run: false   # if it set to True, then environment is rollout infinitely

  ...                   # some other additional args, which is relevant to the environment or some run options
  steps: 200            # it's just for example, get it from an environment or wrapper by
                        # options.get('environment/steps', default_value)
                        # default_value is used if there are no entry at app.yaml 

relaax-rlx-server:
  bind: localhost:7001  # sets address, where environment sends states, rewards (terminals) and receives actions
```

Detailed information about used protocol located there:
- [RELAAX Agent Proxy](../README.md#relaax-agent-proxy)
    - [Reinforcement Learning eXchange protocol](../README.md#reinforcement-learning-exchange-protocol)
    - [Reinforcement Learning eXchange protocol definition](../README.md#reinforcement-learning-exchange-protocol-definition)

It allows to integrate any application of your own choice (see [customized](#customized) section).

Some variants of applications provided in specific [relaax_sample_apps](https://github.com/deeplearninc/relaax_sample_apps) repo.

#### [Basic](#supported-environments)

```bash
$ relaax new -e basic -a [policy-gradient|da3c|trpo|ddpg|dqn] APP_NAME
```

Basic app is provided as some ground sample to test and modify on.

It creates a simple `bandit` app with 4 static predefined slots 'APP_NAME/environment/bandit.py':
```python
class Bandit(object):
    def __init__(self):
        self.slots = [0.2, 0, -0.2, -0.5]

    def pull(self, action):
        result = np.random.randn(1)
        if result > self.slots[action]:
            return 1
        else:
            return -1
```

And operates with the agents trough simple python adapter 'APP_NAME/environment/training.py':
```python
class Training(TrainingBase):
    def __init__(self):
        super(Training, self).__init__()
        self.steps = options.get('environment/steps', 1000)
        self.bandit = Bandit()

    def episode(self, number):
        # get first action from agent
        action = self.agent.update(reward=None, state=[])
        # update agent with state and reward
        for step in range(self.steps):
            reward = self.bandit.pull(action)
            action = self.agent.update(reward=reward, state=[])
            log.info('step: %s, action: %s' % (step, action))
``` 

There are only one additional parameter `step` in default `app.yaml` file,
which controls the amount of steps within one episode:
```yaml
environment:
  run: python environment/training.py  # path to python exchange adapter
  max_episodes: 100                    # how many episodes to run
  infinite_run: false                  # if True, it doesn't stop after `max_episodes` reached
  steps: 300                           # additional parameter to control the episode length
```
<br><br>
s
#### [OpenAI Gym](#supported-environments)

[OpenAI Gym](https://gym.openai.com/) is an open-source library, which provides a collection of
test problems environments, that you can use to work out your reinforcement learning algorithms.

```bash
$ relaax new -e openai-gym -a [policy-gradient|da3c|trpo|ddpg|dqn] APP_NAME
```

Please find sample of configuration to run experiments with OpenAI Gym there:

`relaax/config/da3cc_gym_walker.yaml`

This sample is setup for `BipedalWalker-v2` environment, which operates with continuous action space.
Therefore you may use continuous version of our `Distributed A3C` or set another algorithm there:
```yaml
algorithm:
  path: ../relaax/algorithms/trpo_gae
```

`action_size` and `state_size` parameters for `BipedalWalker-v2` is equal to:
```yaml
action_size: 4                  # action size for the given environment
state_size: [24]                # array of dimensions for the input observation
```
You should check / change these parameter if you want to use another environment.
<br><br>

**How to check action & state sizes by environment name**

Run the `get_info.py` script from `/relaax/environments/OpenAI_Gym`, for example:
```bash
$ python get_info.py Boxing-v0

Action Space: Discrete(18)
Observation Space: Box(210, 160, 3)
Timestep Limit: 10000

$ python get_info.py BipedalWalker-v2

Action Space: Box(4,)
Observation Space: Box(24,)
Timestep Limit: 1600
```
`Timestep Limit` is necessary argument for `trpo-gae` algorithm.

`state_size` for Atari games is equal to `[210, 160, 3]`, which represents an 3-channel
RGB image with `210x160` pixels, but it automatically converts to `[84, 84]`
(1-channel grayscale image of square size) wrt DeepMind's articles.
<br><br>

#### [DeepMind Lab](#supported-environments)

[DeepMind Lab](https://github.com/deepmind/lab) is a 3D learning environment based on
id Software's [Quake III Arena](https://github.com/id-Software/Quake-III-Arena).
It provides a suite of challenging 3D navigation and puzzle-solving tasks
for learning agents especially with deep reinforcement learning.

```bash
$ relaax new -e deepmind-lab -a [policy-gradient|da3c|trpo|ddpg|dqn] APP_NAME
```

Please find sample of configuration to perform experiments with DeepMind Lab there:

`relaax/config/da3c_lab_demo.yaml`

`action_size` and `state_size` parameters for this configuration is equal to:
```yaml
action_size: 3                  # the small action size for the lab's environment
state_size: [84, 84]            # dimensions of the environment's input screen
```

The full set for `action_size` consists of 11-types of interactions.

It allows to define number of desired actions by the 4th parameter.

| Small `action_size`   | Medium `action_size`   | Full `action_size`   |
| ----------------------|:----------------------:| --------------------:|
| look_left             | look_left              | look_left            |
| look_right            | look_right             | look_right           |
| forward               | forward                | forward              |
|                       | strafe_left            | strafe_left          |
|                       | strafe_right           | strafe_right         |
|                       | backward               | backward             |
|                       |                        | look_up              |
|                       |                        | look_down            |
|                       |                        | fire                 |
|                       |                        | jump                 |
|                       |                        | crouch               |
`s` or `small` to set small `action_size`

`m` or `medium` to set medium `action_size` (movement only)

`f` or `full` (`b` or `big`) to set full `action_size`
<br><br>

#### [Customized](#supported-environments)
