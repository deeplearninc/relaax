from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from builtins import range
from builtins import object
import os
import sys
import gym
import random
import numpy as np

from scipy.misc import imresize
from PIL import Image, ImageCms

from gym.spaces import Box
from gym.wrappers.frame_skipping import SkipWrapper
from doom_wrapper import ppaquette_doom

from relaax.client.rlx_client_config import options


class SetFunction(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class GymEnv(object):
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

    ColorSpaces = ['GS', 'CMYK', 'L', 'LAB', 'XYZ']

    def __init__(self, env='CartPole-v0'):
        self._record = options.get('environment/record', False)
        out_dir = options.get('environment/out_dir', '/tmp/'+env)
        if self._record and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if env.startswith('ppaquette/Doom'):
            self.gym = ppaquette_doom(env, self._record, out_dir=out_dir)
        else:
            self.gym = gym.make(env)

            frame_skip = options.get('environment/frame_skip', None)
            if frame_skip is not None:
                skip_wrapper = SkipWrapper(frame_skip)
                self.gym = skip_wrapper(self.gym)

            if self._record:
                self.gym = gym.wrappers.Monitor(self.gym, out_dir, force=True)

        self.gym.seed(random.randrange(1000000))

        self._no_op_max = options.get('environment/no_op_max', 0)
        self._reset_action = self.gym.action_space.sample() \
            if options.get('environment/stochastic_reset', False) else 0

        self._show_ui = options.get('show_ui', False)

        limit = options.get('environment/limit',
                            self.gym.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
        if limit is not None:
            self.gym._max_episode_steps = limit

        self._process_state = SetFunction(self._process_all)

        atari = [name + 'Deterministic' for name in GymEnv.AtariGameList] + GymEnv.AtariGameList
        if any(item.startswith(env.split('-')[0]) for item in atari):
            self._process_state = SetFunction(self._process_atari)

        self.action_size, self.box = self._get_action_size()
        if self.action_size != options.algorithm.output.action_size:
            print('Algorithm expects different action size (%d) from gym (%d). \n'
                  'Please set correct action size in you configuration yaml.' % (
                  options.algorithm.output.action_size, self.action_size))
            sys.exit(-1)

        self.reset()

        self._convert = False
        self._crop = False
        self._shape = False

    def _get_action_size(self):
        space = self.gym.action_space
        if isinstance(space, Box):
            return space.shape[0], True
        return space.n, False

    def act(self, action):
        if self._show_ui or self._record:
            self.gym.render()

        state, reward, terminal, info = self.gym.step(action)

        if terminal:
            state = None
        else:
            state = self._process_state(state)

        return reward, state, terminal

    def reset(self):
        while True:
            state = self.gym.reset()
            terminal = False

            if not self._show_ui and self._no_op_max:
                no_op = np.random.randint(0, self._no_op_max)
                for _ in range(no_op):
                    state, _, terminal, _ = self.gym.step(self._reset_action)

            if not terminal:
                state = self._process_state(state)
                break

        return state

    def _process_img(self, screen):
        if self._convert:
            pass

    @staticmethod
    def _process_atari(screen):  # needs to scale to factor of 42 -> crop: (55, 42)
        gray = np.dot(screen[..., :3], [0.299, 0.587, 0.114])

        resized_screen = imresize(gray, (110, 84))
        state = resized_screen[18:102, :]

        state = state.astype(np.float32)
        state *= (1.0 / 255.0)
        return state

    @staticmethod
    def _process_all(state):
        return state
