from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from builtins import object

import random
import logging
import numpy as np
from PIL import Image

import deepmind_lab
from relaax.environment.config import options

log = logging.getLogger(__name__)


class LabEnv(object):
    def __init__(self):
        self._fps = options.get('environment/fps', 20)
        self._show_ui = options.get('environment/show_ui', False)
        self._no_op_max = options.get('environment/no_op_max', 9)
        self._frame_skip = options.get('environment/frame_skip', 4)
        assert self._fps > 0, log.info('Frame per second rate should be above zero')
        assert self._no_op_max > 0, log.info('Number of random actions at start should be above zero')
        assert self._frame_skip > 0, log.info('Frame skipping rate should be above zero')

        shape = options.get('environment/shape', (84, 84))
        self._height, self._width = shape[0], shape[1]
        self._channels = 1 if len(shape) == 2 else 3

        action_size = options.get('environment/action_size', 'medium')
        if action_size in ACTIONS:
            self._actions = list(ACTIONS[action_size].values())
        else:
            log.info('You\'ve provided an invalid action size. \n'
                     'Valid options are follows: \n {}'.format(ACTIONS.keys()))

        if self._show_ui:
            self._width = 640
            self._height = 480

        self.env = deepmind_lab.Lab(
            options.get('environment/level_script', 'nav_maze_static_01'),
            ['RGB_INTERLACED'],
            config={
                'fps': str(self._fps),
                'width': str(self._width),
                'height': str(self._height)
            })

        self._scale = (1.0 / 255.0)
        self.reset()

    def act(self, action):
        # returns reward, state, terminal
        reward = self.env.step(self._actions[action], num_steps=self._frame_skip)

        terminal = not self.env.is_running()
        if terminal:
            return reward, None, terminal

        state = self._process_state()
        return reward, state, terminal

    def reset(self):
        while True:
            self.env.reset()

            # randomize initial state
            if not self._show_ui and self._no_op_max:
                no_op = np.random.randint(0, self._no_op_max)
                for _ in range(no_op):
                    action = random.choice(self._actions)
                    self.env.step(action, num_steps=self._frame_skip)

            if self.env.is_running():
                break

        state = self._process_state()
        return state

    def _process_state(self):
        screen = self.env.observations()['RGB_INTERLACED']

        if self._channels == 1:
            screen = np.dot(screen[..., :3], [0.299, 0.587, 0.114])

        if self._show_ui:
            screen = np.array(Image.fromarray(screen).resize(
                (self._width, self._height), resample=Image.BILINEAR), dtype=np.uint8)

        processed_screen = screen.astype(np.float32) * self._scale
        return processed_screen


def _action(*entries):
    return np.array(entries, dtype=np.intc)

FULL_ACTIONS = {
    'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
    'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    'look_up': _action(0, 10, 0, 0, 0, 0, 0),
    'look_down': _action(0, -10, 0, 0, 0, 0, 0),
    'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
    'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
    'forward': _action(0, 0, 0, 1, 0, 0, 0),
    'backward': _action(0, 0, 0, -1, 0, 0, 0),
    'fire': _action(0, 0, 0, 0, 1, 0, 0),
    'jump': _action(0, 0, 0, 0, 0, 1, 0),
    'crouch': _action(0, 0, 0, 0, 0, 0, 1)
}

SMALL_ACTIONS = {
    'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
    'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    'forward': _action(0, 0, 0, 1, 0, 0, 0)
}

MEDIUM_ACTIONS = {
    'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
    'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
    'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
    'forward': _action(0, 0, 0, 1, 0, 0, 0),
    'backward': _action(0, 0, 0, -1, 0, 0, 0)
}

ACTIONS = {
    'f': FULL_ACTIONS,
    'full': FULL_ACTIONS,
    'b': FULL_ACTIONS,
    'big': FULL_ACTIONS,
    'm': MEDIUM_ACTIONS,
    'medium': MEDIUM_ACTIONS,
    's': SMALL_ACTIONS,
    'small': SMALL_ACTIONS
}
