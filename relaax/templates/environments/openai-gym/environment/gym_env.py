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
from relaax.common.rlx_message import RLXMessageImage

gym.configuration.undo_logger_setup()
log = logging.getLogger(__name__)


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

        shape = options.get('environment/shape', options.get('environment/image', (84, 84)))
        self._shape = shape[:2]
        if len(self._shape) > 1:
            self._channels = 0 if len(self._shape) == 2 else self._shape[-1]

        self._crop = options.get('environment/crop', True)
        self._process_state = self._process_all

        atari = [name + 'Deterministic' for name in GymEnv.AtariGameList] + GymEnv.AtariGameList
        if any(item.startswith(env.split('-')[0]) for item in atari):
            self._process_state = self._process_img

        self.action_size = self._get_action_size()
        if self.action_size != options.algorithm.output.action_size:
            log.error('Algorithm expects action size %d; gym return %d. \n'
                      'Please set correct action size in you configuration yaml.' %
                      (options.algorithm.output.action_size, self.action_size))
            sys.exit(-1)

        self._scale = (1.0 / 255.0)
        self.reset()

    def _get_action_size(self):
        space = self.gym.action_space
        if isinstance(space, Box):
            return space.shape[0]
        return space.n

    def get_action_high(self):
        return self.gym.action_space.high

    def act(self, action):
        if self._show_ui or self._record:
            self.gym.render()

        state, reward, terminal, info = self.gym.step(action)
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
        if self._channels < 2:
            screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        if self._crop:
            screen = screen[34:34 + 160, :160]

        if self._shape[0] < 80:
            screen = np.array(Image.fromarray(screen).resize((80, 80), resample=Image.BILINEAR),
                              dtype=np.uint8)

        screen = RLXMessageImage(Image.fromarray(screen).resize(self._shape, resample=Image.BILINEAR))

        return screen

    def _process_all(self, state):
        if self._shape == (84, 84):
            self._shape = state.shape
        if state.shape != self._shape:
            state = np.reshape(state, self._shape)
        return state
