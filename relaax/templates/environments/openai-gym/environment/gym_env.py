from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from builtins import range
from builtins import object

import os
import sys
import gym
import random
import logging
import numpy as np
from PIL import Image

from gym.spaces import Box
from gym.wrappers.frame_skipping import SkipWrapper

from relaax.environment.config import options

gym.configuration.undo_logger_setup()
log = logging.getLogger(__name__)


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

    def __init__(self, env='CartPole-v0'):
        self.gym = gym.make(env)

        frame_skip = options.get('environment/frame_skip', None)
        if frame_skip is not None:
            skip_wrapper = SkipWrapper(frame_skip)
            self.gym = skip_wrapper(self.gym)

        self._record = options.get('environment/record', False)
        if self._record:
            out_dir = options.get('environment/out_dir', '/tmp/' + env)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            self.gym = gym.wrappers.Monitor(self.gym, out_dir, force=True)

        self._no_op_max = options.get('environment/no_op_max', 0)
        self._reset_action = self.gym.action_space.sample() \
            if options.get('environment/stochastic_reset', False) else 0

        self.gym.seed(random.randrange(1000000))
        self._show_ui = options.get('show_ui', False)

        limit = options.get('environment/limit',
                            self.gym.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
        if limit is not None:
            self.gym._max_episode_steps = limit

        shape = options.get('environment/shape', (84, 84))
        self._shape = (shape[0], shape[1])
        self._channels = 1 if len(shape) == 2 else 3

        if options.get('environment/crop', True):
            self._crop = True
            self._top = round(9 * (shape[0] / 42))
            self._bottom = round(shape[0] - 4 * (shape[0] / 42))

        self._process_state = SetFunction(self._process_all)

        atari = [name + 'Deterministic' for name in GymEnv.AtariGameList] + GymEnv.AtariGameList
        if any(item.startswith(env.split('-')[0]) for item in atari):
            self._process_state = SetFunction(self._process_img)

        self.action_size = self._get_action_size()
        if self.action_size != options.algorithm.output.action_size:
            print('Algorithm expects different action size (%d) from gym (%d). \n'
                  'Please set correct action size in you configuration yaml.' % (
                   options.algorithm.output.action_size, self.action_size))
            sys.exit(-1)

        self._scale = (1.0 / 255.0)
        self.reset()

    def _get_action_size(self):
        space = self.gym.action_space
        if isinstance(space, Box):
            return space.shape[0]
        return space.n

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
        if self._channels == 1:
            screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114])

        if self._crop:
            screen = screen[self._top:self._bottom, ...]

        screen = np.array(Image.fromarray(screen).resize(
            self._shape, resample=Image.BILINEAR), dtype=np.uint8)

        # return processed screen
        return screen.astype(np.float32) * self._scale

    @staticmethod
    def _process_all(state):
        return state
