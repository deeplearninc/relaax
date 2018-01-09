from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from builtins import range
from builtins import object

import os
import sys
import gym
import time
import random
import logging
import numpy as np
from collections import deque
from PIL import Image

from gym.spaces import Box
from gym.wrappers.frame_skipping import SkipWrapper
from ppaquette_gym_doom import wrappers

from relaax.environment.config import options
from relaax.common.rlx_message import RLXMessageImage

gym.configuration.undo_logger_setup()
log = logging.getLogger(__name__)


class DoomEnv(object):
    def __init__(self, level='ppaquette/DoomMyWayHome-v0'):
        time.sleep(np.random.randint(100))
        env = gym.make(level)

        modewrapper = wrappers.SetPlayingMode('algo')
        obwrapper = wrappers.SetResolution('160x120')
        acwrapper = wrappers.ToDiscrete('minimal')
        env = modewrapper(obwrapper(acwrapper(env)))

        frame_skip = options.get('environment/frame_skip', 4)
        if frame_skip is not None:
            skip_wrapper = SkipWrapper(frame_skip)
            env = skip_wrapper(env)

        self._record = options.get('environment/record', False)
        if self._record:
            out_dir = options.get('environment/out_dir', '/tmp/' + level.split('/')[-1])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            env = gym.wrappers.Monitor(env, out_dir, force=True)

        self._no_op_max = options.get('environment/no_op_max', 0)
        self._reset_action = env.action_space.sample() \
            if options.get('environment/stochastic_reset', False) else 0

        env.seed(random.randrange(1000000))
        self._show_ui = options.get('show_ui', False)

        limit = options.get('environment/limit',
                            env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
        if limit is not None:
            env._max_episode_steps = limit

        shape = options.get('environment/shape', (42, 42))
        self._shape = shape[:2]
        assert len(self._shape) > 1, 'You should provide 2D or 3D shape'
        self._channels = 0 if len(self._shape) == 2 else self._shape[-1]

        self.action_size = self._get_action_size(env)
        if self.action_size != options.algorithm.output.action_size:
            print('Algorithm expects different action size (%d) from gym (%d). \n'
                  'Please set correct action size in you configuration yaml.' % (
                   options.algorithm.output.action_size, self.action_size))
            sys.exit(-1)

        if options.get('environment/no_life', True):
            env = NoNegativeRewardEnv(env)
        self.env = env

        self._process_img = self._process_common
        if options.get('environment/max_pool', True):
            self._obs_buffer = deque(maxlen=2)
            self._process_img = self._process_max

        self.observation_space = Box(0.0, 255.0, shape)
        self.observation_space.high[...] = 1.0

        self.reset()

    def _get_action_size(self, env):
        space = env.action_space
        if isinstance(space, Box):
            return space.shape[0]
        return space.n

    def act(self, action):
        if self._show_ui or self._record:
            self.env.render()

        state, reward, terminal, info = self.env.step(action)
        state = self._process_img(state)

        return reward, state, terminal

    def reset(self):
        self._obs_buffer.clear()

        while True:
            state = self.env.reset()
            terminal = False

            if not self._show_ui and self._no_op_max:
                no_op = np.random.randint(0, self._no_op_max)
                for _ in range(no_op):
                    state, _, terminal, _ = self.env.step(self._reset_action)

            if not terminal:
                state = self._process_img(state)
                break

        return state

    def _process_common(self, screen):
        if self._channels < 2:
            screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        screen = RLXMessageImage(Image.fromarray(screen).resize(self._shape, resample=Image.BILINEAR))

        return screen

    def _process_max(self, screen):
        self._obs_buffer.append(screen)
        screen = np.max(np.stack(self._obs_buffer), axis=0)

        if self._channels < 2:
            screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        screen = RLXMessageImage(Image.fromarray(screen).resize(self._shape, resample=Image.BILINEAR))

        return screen


class NoNegativeRewardEnv(gym.RewardWrapper):
    """Clip reward in negative direction."""
    def __init__(self, env=None, neg_clip=0.0):
        super(NoNegativeRewardEnv, self).__init__(env)
        self.neg_clip = neg_clip

    def _reward(self, reward):
        new_reward = self.neg_clip if reward < self.neg_clip else reward
        return new_reward
