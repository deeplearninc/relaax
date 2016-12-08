# Client for OpenAI's gym Environments
 
 OpenAI's gym allows you to launch different types of environments,
 not even Atari games, for example:
 
 * [Classic Control](https://gym.openai.com/envs#classic_control)
 * [Doom](https://gym.openai.com/envs#doom)
 * [Algorithmic](https://gym.openai.com/envs#algorithmic)
 * [PyGame Learning Environment](https://gym.openai.com/envs#pygame)
 * [Walkers, Landers & Racing](https://gym.openai.com/envs##box2d)
 * ant others, see the full list [there](https://gym.openai.com/envs)
 
 To run client just evaluate `main.py`
 which already has default parameters or change some via command line.
 
 Parameters, which you can specify:
 
 [1] --algo   `name_of_algorithm`           `[default = 'a3c']`
 
 You can choose between: `a3c`, `dqn`, `q_learn`
 For `a3c` you can also specify `lstm` usage `[default = True]`
 if you set this to `False` the `feed-forward` would be used instead
 
 [2] --env    `environment_name`             `[default = 'Boxing-v0']`
 
 See full list of supported environments [there](https://gym.openai.com/envs)
 
 [3] --agents `number_of_parallel_agents`   `[default = 8]`
 
 [4] --lstm `False` if you want to use FF `[default = True]`
 
 For example, we can change the number of threads for default
 `Asynchronous Advanced Actor-Critic` algorithm:
 > main.py --agents 16
 
 You can also specify another `game name` and `algorithm`:
 > main.py --game pong --algo dqn
 
 If you want to see some visual output for one game just type in terminal `d` then `Enter`
 (if game didn't start try to repeat one more time)
 
## How to Run & Dependencies

Before you start, make sure you have installed on your system:

- `python 2.7 or 3.5`

- [`pip`](https://pip.pypa.io/en/stable/installing/) - just need to install requirements, see command below:
    > pip install -r requirements.txt

If installation of `gym` failed try to install these dependencies:

`sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig`

Then install `gym` again:

 > pip install gym

And for full installation:

 > pip install 'gym[all]'
 
#### [How to create a virtual environment](/VirtualEnvironments.md)