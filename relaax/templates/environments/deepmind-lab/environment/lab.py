from __future__ import division
from builtins import str
from builtins import range
from builtins import object
import random
import numpy as np
from scipy.misc import imresize
import deepmind_lab

from relaax.client.rlx_client_config import options


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class LabEnv(object):
    ACTIONS = {
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
    ACTION_LIST = list(ACTIONS.values())

    CONVERT = {0: 0, 1: 9, 2: 10, 3: 3, 4: 9, 5: 0, 6: 6, 7: 6, 8: 8, 9: 9, 10: 10}

    def __init__(self):
        self._fps = options.get('environment/fps', 20)
        self._width = options.get('environment/width', 84)
        self._height = options.get('environment/height', 84)
        self._show_ui = options.get('environment/show_ui', False)
        self._shrink = options.get('environment/shrink', True)
        self._no_op_max = options.get('environment/no_op_max', 7)
        self._frame_skip = options.get('environment/frame_skip', 4)

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

        self.reset()

    def act(self, action):
        if self._shrink:
            action = LabEnv.CONVERT[action]
        # returns reward, state, terminal
        return self._process_frame(LabEnv.ACTION_LIST[action])

    def _process_frame(self, action):
        reward = self.env.step(action, num_steps=self._frame_skip)
        terminal = not self.env.is_running()

        if terminal:
            return reward, None, terminal

        # train screen shape is (84, 84) by default
        screen = self.env.observations()['RGB_INTERLACED']
        x_t = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
        if self._show_ui:
            x_t = imresize(x_t, (self._width, self._height))

        x_t = x_t.astype(np.float32)
        x_t *= (1.0 / 255.0)
        return reward, x_t, terminal

    def reset(self):
        while True:
            self.env.reset()

            # randomize initial state
            if self._no_op_max > 0:
                no_op = np.random.randint(0, self._no_op_max + 1)
                for _ in range(no_op):
                    action = random.choice(LabEnv.ACTION_LIST)
                    self.env.step(action, num_steps=self._frame_skip)

            _, _, terminal = self._process_frame(random.choice(LabEnv.ACTION_LIST))
            if not terminal:
                break
