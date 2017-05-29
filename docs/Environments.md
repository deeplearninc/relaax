### [Supported Environments](../README.md#contents)
> click on title to go to contents
- [ALE](#arcade-learning-environment)
- [OpenAI Gym](#openai-gym)
- [DeepMind Lab](#deepmind-lab)

#### [Arcade-Learning-Environment](#supported-environments)

The [Arcade Learning Environment (ALE)](http://www.arcadelearningenvironment.org/)
is a framework that allows to develop AI agents for Atari 2600 games.
It is built on top of the Atari 2600 emulator [Stella](http://stella.sourceforge.net/)
and separates the details of emulation from agent design. Additional information
about ALE and Atari games you can find in official [Google group.](https://groups.google.com/forum/#!forum/arcade-learning-environment)

1. Pull the Docker Image:

    ```bash
    $ docker pull deeplearninc/relaax-ale:v0.2.0
    ```

2. Run the Server:

    Open new terminal window, navigate to training directory and run `honcho`:
    ```bash
    $ honcho -f ../relaax/config/da3c_ale_boxing.Procfile start
    ```
    It is assumed that the training directory located next to `relaax` repository
    at the same level. It also allows to create it anywhere and it needs
    to write the right path to the appropriate `*.Procfile` within `relaax` repo.

3. Run a Client:

    It provides some run-cases for the pulled docker image:
    ```bash
    # For example, the first one case

    $ docker run --rm -ti \
        -v /path_to_atari_roms_folder:/roms \
        --name ale deeplearninc/relaax-ale:v0.2.0 \
        --rlx-server SERVER_IP:PORT --env boxing
    ```
    It runs the docker in interactive mode by `-ti` and automatically removes this
    container when it stops with `--rm`. It also has `--name ale` for convenience.

    You have to provide shared folder on your computer, where atari game roms are
    stored by `-v` parameter.

    Use `ifconfig` command to find IP of your relaax SERVER, which is run by `honcho`

    It launches one sample of the game environment within the docker, which is defined
    by the last parameter `boxing` (it launches the `Atari Boxing` game)

    ```bash
    # For example, another run-case

    $ docker run --rm -ti \
        -v /path_to_atari_roms_folder:/roms \
        --name ale deeplearninc/relaax-ale:v0.2.0 \
        --rlx-server SERVER_IP:PORT --env boxing -n 4
    ```
    It adds the third parameter `-n` which is equal to `4` since it allows to
    define number of games to launch within the docker for parallel training.

    ```bash
    # Use-case with dispaly mode

    $ docker run --rm -ti \
        -p IP:PORT:5900 \
        -v /path_to_atari_roms_folder:/roms \
        --name ale deeplearninc/relaax-ale:v0.2.0 \
        --rlx-server SERVER_IP:PORT --env boxing --display
    ```
    It passes the last argument as `display` to run game in display mode, therefore
    it maps some ports on your computer to use `VNC` connection for visual session.

    For example, the full command to run the clients and a server on
    a single machine (under the NAT) should looks like as follows:
    ```bash
    $ docker run --rm -d \
        -p 192.168.2.103:15900:5900 \
        -v /opt/atari-game-roms:/roms \
        --name ale deeplearninc/relaax-ale:v0.2.0 \
        -x 192.168.2.103:7001 -e boxing -d
    ```
    It runs docker in a background mode by `-d` instead of interactive.
    It also uses short names such as `-x`, `-e` and `-d` which replaces
    `--rlx-server`, `--env` and `--display` respectively.

    You can connect to client's visual output via your VNC client with:
    ```
    For example:
    ---
    Server: 192.168.2.103:15900
    Passwd: relaax
    Color depth: True color (24 bit)
    ```

    You also can launch docker directly in `localhost` mode on a single PC:
    ```bash
    It works for *nix OS:
    ---
    $ docker run --rm -d \
        --net host \
        --name ale deeplearninc/relaax-ale:v0.2.0 \
        --rlx-server localhost:7001 \
        --env breakout --display
    ```
    It allows to set the server in your VNC client as `localhost:5900`

Please find sample of configuration to perform experiments with ALE there:

`relaax/config/da3c_ale_boxing.yaml`

This sample is setup for `Atari Boxing` game, which has a discrete set of actions.
Therefore you may use discrete version of our `Distributed A3C` or set another algorithm there:
```yml
algorithm:
  path: ../relaax/algorithms/da3c
```

`action_size` and `state_size` parameters for `Atari Boxing` is equal to:
```yml
action_size: 18                 # action size for given game rom (18 fits ale boxing)
state_size: [84, 84]            # dimensions of input screen frame of an Atari game
```
You should check / change these parameter if you want to use another environment.
<br><br>

**How to build your own Docker Image**

* Navigate to the ALE's folder within `relaax` repo
```bash
$ cd path_to_relaax_repo/environments/ALE
```

* Build the docker image by the following commands
```bash
# docker build -f Dockerfile -t your_docker_hub_name/image_name ../..
# or you can build without your docker hub username, for example:

$ docker build -f Dockerfile -t relaax-ale-vnc ../..
```
<br><br>

#### [OpenAI Gym](#supported-environments)

[OpenAI Gym](https://gym.openai.com/) is open-source library: a collection of test problems environments,
that you can use to work out your reinforcement learning algorithms.

1. Pull the Docker Image:

    ```bash
    $ docker pull deeplearninc/relaax-gym:v0.2.0
    ```

2. Run the Server:

    Open new terminal window, navigate to training directory and run `honcho`:
    ```bash
    $ honcho -f ../relaax/config/trpo_gym_walker.Procfile start
    ```
    It is assumed that the training directory located next to `relaax` repository
    at the same level. It also allows to create it anywhere and it needs
    to write the right path to the appropriate `*.Procfile` within `relaax` repo.

3. Run a Client:

    Let's explain some run-cases for the pulled docker image:
    ```bash
    # For example, the first one case

    $ docker run --rm -ti \
        --name gym deeplearninc/relaax-gym:v0.2.0 \
        --rlx-server SERVER_IP:PORT --env BipedalWalker-v2
    ```
    It launches one sample of the environment within the docker, which is defined
    by the last parameter `--env` with name `BipedalWalker-v2`
    You can find more environment names [there](https://gym.openai.com/envs)

    The command above runs the docker in interactive mode by `-ti` and automatically removes
    this container when it stops with `--rm`. It also has `--name gym` for convenience.
    You can stop the docker by pressing `Ctrl+C` in docker's cmd at any time.

    But I recommend to use `-d` parameter instead of `-ti` which allows to launch
    docker is a background mode. You can stop the docker further by:
    ```bash
    $ docker stop gym
    ```

    Use `ifconfig` command to find IP of your relaax SERVER, which is run by `honcho`

    ```bash
    # One more run-case

    $ docker run --rm -ti \
        --name gym deeplearninc/relaax-gym:v0.2.0 \
        --rlx-server SERVER_IP:PORT --env BipedalWalker-v2 -n 4
    ```
    It adds the third parameter `-n` which is equal to `4` since it allows to define
    number of environments to launch within the docker for parallel training.

    ```bash
    # Use-case with visual output

    $ docker run --rm -ti \
        -p IP:PORT:5900 \
        --name gym deeplearninc/relaax-gym:v0.2.0 \
        --rlx-server SERVER_IP:PORT --env BipedalWalker-v2 --display
    ```
    It passes the last argument `--display` to run environment in a display mode, therefore
    it maps some ports on your computer to use `VNC` connection for visual session.

    For example, the full command to run the clients and a server on
    a single machine (under the NAT) should looks like as follows:
    ```bash
    $ docker run --rm -ti \
        -p 192.168.2.103:15900:5900 \
        --name gym deeplearninc/relaax-gym:v0.2.0 \
        --rlx-server 192.168.2.103:7001 \
        --env BipedalWalker-v2 --display
    ```

    You can connect to client's visual output via your VNC client with:
    ```
    For example:
    ---
    Server: 192.168.2.103:15900
    Passwd: relaax
    Color depth: True color (24 bit)
    ```

    You also can launch docker directly in `localhost` mode for single PC:
    ```bash
    It works for *nix OS:
    ---
    $ docker run --rm -d \
        --net host \
        --name gym deeplearninc/relaax-gym:v0.2.0 \
        --rlx-server localhost:7001 \
        --env BipedalWalker-v2 --display
    ```
    It allows to set the server in your VNC client as `localhost:5900`

Please find sample of configuration to run experiments with OpenAI Gym there:

`relaax/config/da3cc_gym_walker.yaml`

This sample is setup for `BipedalWalker-v2` environment, which operates with continuous action space.
Therefore you may use continuous version of our `Distributed A3C` or set another algorithm there:
```yml
algorithm:
  path: ../relaax/algorithms/trpo_gae
```

`action_size` and `state_size` parameters for `BipedalWalker-v2` is equal to:
```yml
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

**How to build your own Docker Image**

* Navigate to the OpenAI Gym's folder within `relaax` repo
```bash
$ cd path_to_relaax_repo/environments/OpenAI_Gym
```

* Build the docker image by the following commands
```bash
# docker build -f Dockerfile -t your_docker_hub_name/image_name ../..
# or you can build without your docker hub username, for example:

$ docker build -f Dockerfile -t relaax-gym-vnc ../..
```
<br><br>

#### [DeepMind Lab](#supported-environments)

[DeepMind Lab](https://github.com/deepmind/lab) is a 3D learning environment based on
id Software's [Quake III Arena](https://github.com/id-Software/Quake-III-Arena).
It provides a suite of challenging 3D navigation and puzzle-solving tasks
for learning agents especially with deep reinforcement learning.

1. Pull the Docker Image:

    ```bash
    $ docker pull deeplearninc/relaax-lab:v0.2.0
    ```

2. Run the Server:

    Open new terminal window, navigate to training directory and run `honcho`:
    ```bash
    $ honcho -f ../relaax/config/da3c_lab_demo.Procfile start
    ```
    It is assumed that the training directory located next to `relaax` repository
    at the same level. It also allows to create it anywhere and it needs
    to write the right path to the appropriate `*.Procfile` within `relaax` repo.

3. Run a Client:

    It provides 3 predefined run-cases for the pulled docker image:
    ```bash
    # For example, the first one case

    $ docker run --rm -ti \
        --name lab deeplearninc/relaax-lab:v0.2.0 \
        SERVER_IP:PORT
    ```
    It runs the docker in interactive mode by `-ti` and automatically removes this
    container when it stops with `--rm`. It also has `--name lab` for convenience.

    Use `ifconfig` command to find IP of your relaax SERVER, which is run by `honcho`

    It launches one sample of the lab's environment within the docker with a `nav_maze_static_01` map
    which is predefined by default (list of the default lab's [maps](https://github.com/deepmind/lab/tree/master/assets/maps))

    ```bash
    # For example, the second run-case

    $ docker run --rm -ti \
        --name lab deeplearninc/relaax-lab:v0.2.0 \
        SERVER_IP:PORT 4 nav_maze_static_02 full
    ```
    It adds the second parameter which is equal to `4` since it allows to define
    number of environments to launch within the docker for parallel training.

    It also allows to define a `map` by the third parameter or it uses `nav_maze_static_01` by default;
    and an `action size`, which set to `full` in this case (see explanation below, `m` by default).

    ```bash
    # And the third one use-case

    $ docker run --rm -ti \
        -p IP:PORT:6080 \
        --name lab deeplearninc/relaax-lab:v0.2.0 \
        SERVER_IP:PORT display
    ```
    It passes the last argument as `display` to run environment in display mode, therefore
    it maps some ports on your computer to use `VNC` connection for visual session.

    It also allows to define a `map` and `action size` by the 3rd and 4th parameter respectively.

    For example, the full command to run the clients and a server on
    a single machine (under the NAT) should looks like as follows:
    ```bash
    $ docker run --rm -ti \
        -p 6080:6080 \
        --name lab deeplearninc/relaax-lab:v0.2.0 \
        192.168.2.103:7001 display nav_maze_static_03 s
    ```

    You can connect to client's visual output via your browser by opening http://127.0.0.1:6080/vnc.html URL.
    You will see web form to enter your credentials. Leave all fields intact and press `'Connect'`.
    You will see a running game.

Please find sample of configuration to perform experiments with DeepMind Lab there:

`relaax/config/da3c_lab_demo.yaml`

`action_size` and `state_size` parameters for this configuration is equal to:
```yml
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

**How to build your own Docker Image**

* Navigate to the DeepMind Lab's folder within `relaax` repo
```bash
$ cd path_to_relaax_repo/environments/DeepMind_Lab
```

* Build the docker image by the following commands
```bash
# docker build -f Dockerfile -t your_docker_hub_name/image_name ../..
# or you can build without your docker hub username, for example:

$ docker build -f Dockerfile -t relaax-lab-vnc ../..
```