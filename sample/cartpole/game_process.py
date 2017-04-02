import numpy as np
from scipy.misc import imresize

import gym    # you should install gym via pip
from gym.spaces import Box  # check continuous
import random


class GameProcessFactory(object):
    def __init__(self):
        pass

    def new_env(self):
        return _GameProcess()

    def new_display_env(self):
        return _GameProcess(display=True, no_op_max=0)


class SetFunction(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class _GameProcess(object):
    AtariGameList = [
        'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix',
        'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider',
        'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
        'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
        'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite',
        'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond', 'JourneyEscape',
        'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
        'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan',
        'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
        'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner',
        'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture',
        'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']

    def __init__(self, env='CartPole-v0', display=False, no_op_max=0, limit=800):
        self.gym = gym.make(env)

        self.gym.seed(random.randrange(1000000))
        self._no_op_max = no_op_max

        self.display = display
        self._close_display = False

        self.timestep_limit = limit
        if self.timestep_limit is None:
            self.timestep_limit = self.gym.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        self.cur_step_limit = None
        self._state = None

        self._process_state = SetFunction(self._process_all)
        self.reset = SetFunction(self._reset_all)

        atari = [name + 'Deterministic' for name in _GameProcess.AtariGameList] + _GameProcess.AtariGameList

        if any(item.startswith(env.split('-')[0]) for item in atari):
            self._process_state = SetFunction(self._process_atari)
            self.reset = SetFunction(self._reset_atari)

        self.ac_size, self.box = self._action_size()
        self.reset()

    def _action_size(self):
        space = self.gym.action_space
        if isinstance(space, Box):
            return space.shape[0], True
        return space.n, False

    def state(self):
        return self._state

    def act(self, action):
        # if self.box:
        #     action = np.clip(action, self.gym.action_space.low, self.gym.action_space.high)
        if self.display:
            if self._close_display:
                self.gym.render(close=True)
                self.display = False
                self._close_display = False
            else:
                self.gym.render()

        state, reward, terminal, info = self.gym.step(action)
        self._state = self._process_state(state)

        self.cur_step_limit += 1
        if self.cur_step_limit > self.timestep_limit:
            terminal = True

        return reward, terminal

    def _reset_atari(self):
        while True:
            self.gym.reset()
            self.cur_step_limit = 0

            if not self.display and self._no_op_max:
                no_op = np.random.randint(0, self._no_op_max)
                # self.cur_step_limit += no_op

                for _ in range(no_op):
                    self.gym.step(0)

            env_state = self.gym.step(0)
            if not env_state[2]:  # not terminal
                self._state = self._process_state(env_state[0])
                # self.cur_step_limit += 1
                break

    def _reset_all(self):
        while True:
            self.gym.reset()
            self.cur_step_limit = 0

            if not self.display and self._no_op_max:
                no_op = np.random.randint(0, self._no_op_max)
                # self.cur_step_limit += no_op

                for _ in range(no_op):
                    self.gym.step(self.gym.action_space.sample())

            env_state = self.gym.step(self.gym.action_space.sample())
            if not env_state[2]:  # not terminal
                self._state = self._process_state(env_state[0])
                # self.cur_step_limit += 1
                break

    @staticmethod
    def _process_atari(screen):
        gray = np.dot(screen[..., :3], [0.299, 0.587, 0.114])

        resized_screen = imresize(gray, (110, 84))
        state = resized_screen[18:102, :]

        state = state.astype(np.float32)
        state *= (1.0 / 255.0)
        return state

    @staticmethod
    def _process_all(state):
        return state

    def _safe_rnd_act(self):
        act = (np.random.randn(1, self.ac_size).astype(np.float32))[0]
        return np.clip(act, self.gym.action_space.low, self.gym.action_space.high)
