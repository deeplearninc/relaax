from __future__ import division
from builtins import range
from builtins import object
import imp
import os
import sys
import numpy as np
from scipy.misc import imresize

import ale_python_interface


class GameProcessFactory(object):
    def __init__(self, rom, display):
        self._rom = rom
        self._display = display

    def new_env(self, seed):
        return _GameProcess(seed, self._rom, self._display)

    def new_display_env(self, seed):
        return _GameProcess(seed, self._rom, display=True, no_op_max=0)


class _GameProcess(object):
    def __init__(self, rand_seed, rom, display=False, frame_skip=4, no_op_max=7):
        self.ale = ale_python_interface.ALEInterface()

        self.ale.setInt(b'random_seed', rand_seed)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setInt(b'frame_skip', frame_skip)

        self._no_op_max = no_op_max
        if display:
            self._setup_display()

        self.ale.loadROM(rom.encode('ascii'))

        # collect minimal action set
        self.real_actions = self.ale.getMinimalActionSet()

        # height=210, width=160
        self._screen = np.empty((210, 160, 1), dtype=np.uint8)

        self.reset()

    def action_size(self):
        return len(self.ale.getMinimalActionSet())

    def state(self):
        return self.s_t

    def act(self, action):
        # convert original 18 action index to minimal action set index
        real_action = self.real_actions[action]
        self.reward, self.terminal, self.s_t = self._process_frame(real_action)
        return self.reward, self.terminal

    def _process_frame(self, action):
        reward = self.ale.act(action)
        terminal = self.ale.game_over()

        # screen shape is (210, 160, 1)
        self.ale.getScreenGrayscale(self._screen)

        # reshape it into (210, 160)
        reshaped_screen = np.reshape(self._screen, (210, 160))

        # resize to height=110, width=84
        resized_screen = imresize(reshaped_screen, (110, 84))

        x_t = resized_screen[18:102, :]
        x_t = x_t.astype(np.float32)
        x_t *= (1.0 / 255.0)
        return reward, terminal, x_t

    def _setup_display(self):
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            self.ale.setBool('sound', False)
        elif sys.platform.startswith('linux'):
            self.ale.setBool('sound', True)
        self.ale.setBool('display_screen', True)

    def reset(self):
        self.ale.reset_game()

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.ale.act(0)

        _, _, self.s_t = self._process_frame(0)

        self.reward = 0
        self.terminal = False


def _load_module(path, name):
    if name not in sys.modules:
        file, pathname, description = imp.find_module(name, [path])
        try:
            imp.load_module(name, file, pathname, description)
        finally:
            if file:
                file.close()
    return sys.modules[name]
