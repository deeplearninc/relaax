import os
import sys
import numpy as np
from scipy.misc import imresize

from ale_python_interface import ALEInterface


class GameProcess(object):
    def __init__(self, rand_seed, game_name, display=False, frame_skip=4, no_op_max=7):
        self.ale = ALEInterface()

        self.ale.setInt(b'random_seed', rand_seed)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setInt(b'frame_skip', frame_skip)

        self._no_op_max = no_op_max
        if display:
            self._setup_display()

        ROM = os.path.dirname(__file__) + '/atari-games/' + game_name + '.bin'
        self.ale.loadROM(ROM.encode('ascii'))

        # collect minimal action set
        self.real_actions = self.ale.getMinimalActionSet()

        # height=210, width=160
        self._screen = np.empty((210, 160, 1), dtype=np.uint8)

        self.reset()

    def _process_frame(self, action, reshape):
        reward = self.ale.act(action)
        terminal = self.ale.game_over()

        # screen shape is (210, 160, 1)
        self.ale.getScreenGrayscale(self._screen)

        # reshape it into (210, 160)
        reshaped_screen = np.reshape(self._screen, (210, 160))

        # resize to height=110, width=84
        resized_screen = imresize(reshaped_screen, (110, 84))

        x_t = resized_screen[18:102, :]
        if reshape:
            x_t = np.reshape(x_t, (84, 84, 1))

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

        _, _, self.s_t = self._process_frame(0, False)

        self.reward = 0
        self.terminal = False

    def process(self, action):
        # convert original 18 action index to minimal action set index
        real_action = self.real_actions[action]

        r, t, self.s_t = self._process_frame(real_action, True)

        self.reward = r
        self.terminal = t

    def real_action_size(self):
        return len(self.ale.getMinimalActionSet())
