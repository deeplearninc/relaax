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

And operates with the agents trough simple python adapter `APP_NAME/environment/training.py`:
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

#### [OpenAI Gym](#supported-environments)

[OpenAI Gym](https://gym.openai.com/) is an open-source library, which provides a collection of
test problems environments, that you can use to work out your reinforcement learning algorithms.

```bash
$ relaax new -e openai-gym -a [policy-gradient|da3c|trpo|ddpg|dqn] APP_NAME
```

And operates with the agents trough adapter `APP_NAME/environment/training.py`, which looks as follows:
```python
from gym_env import GymEnv

class Training(TrainingBase):
    def __init__(self):
        super(Training, self).__init__()
        self.gym = GymEnv(env=options.get('environment/name', 'CartPole-v0'))

    def episode(self, number):
        state = self.gym.reset()
        reward, episode_reward, terminal = None, 0, False
        action = self.agent.update(reward, state, terminal)
        while not terminal:
            reward, state, terminal = self.gym.act(action)
            action = self.agent.update(reward, state, terminal)
            episode_reward += reward
        log.info('Episode %d reward %d' % (number, episode_reward))
        return episode_reward
```

It also use specific wrapper for any Gym environment, which is located there `APP_NAME/environment/gym_env.py`  
It allows to setup given Gym environment by some options at your `app.yaml` file:
Please find sample of configuration to run experiments with OpenAI Gym there:
```yaml
environment:
  run: python environment/training.py  # path to gym's wrapper to run
  name: PongDeterministic-v4           # gym's environment name 
  shape: [42, 42]                      # output shape of the environment's state
  max_episodes: 10000                  # number of episodes to run
  infinite_run: True                   # if True, it doesn't stop after `max_episodes` reached

  frame_skip: 4                        # number of frames to skip between consecutive states,
                                       # 4 is default one for most Atari games
  record: False                        # if True, it uses the default gym's monitor to record
  out_dir: /tmp/pong                   # folder to store monitor's files 
  no_op_max: 30                        # maximum number of random actions before take the control
  stochastic_reset: False              # if False, it uses 0-th action for no_op_max, instead ramdomly
  show_ui: False                       # set to True to see the environment's UI
  limit: 100000                        # number of steps to pass trough one episode, uses gym's defaults instead
  crop: True                           # use screen cropping for Atari games to restrict the game area
```
All options can be omitted (except `run`) and it uses the default ones instead.
<br><br>

**How to check action & state sizes by Gym's environment name**

It necessary to know the `size` of `actions & states` for the given environment.  
There are some simple script to check these information:  
(it's assumed that the `gym` is already installed) 
```python
import sys
import gym

env = gym.make(sys.argv[1])

print('\nAction Space:', env.action_space)
print('Observation Space:', env.observation_space)
print('Timestep Limit:', env.spec.timestep_limit, '\n')
```

Just run this script from terminal and it should output something like this:
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
`Timestep Limit` is necessary argument for `trpo` algorithm, but it allows to set any one by `limit` parameter.

`state_size` for Atari games is equal to `[210, 160, 3]`, which represents an 3-channel
RGB image with `210x160` pixels, but it could be converted to any shape wrt `app.yaml`.
For example, if at `app.yaml` file the shape is set to `[84, 84]` (environment/shape).
It automatically converts the source RGB to 1-channel grayscale image of defined size.
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
