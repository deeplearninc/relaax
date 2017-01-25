import deepmind_lab
import numpy as np
import random

from scipy.misc import imresize


class GameProcessFactory(object):
    def __init__(self, level, width, height, display, action_size):
        self._level = level
        self._width = width
        self._height = height
        self._display = display
        self._action_size = action_size

    def new_env(self, seed, frame_skip):
        return _GameProcess(seed, self._level, self._width, self._height, frame_skip, self._display, self._action_size)

    def new_display_env(self, seed, frame_skip):
        return _GameProcess(seed, self._level, self._width, self._height, frame_skip, display=True, no_op_max=0)


class _GameProcess(object):

    def __init__(self, fps, level, width, height, frame_skip, display=False, action_size='m', no_op_max=0):
        self._frame_skip = frame_skip
        self._no_op_max = no_op_max
        self._width = width
        self._height = height
        self._display = display
        self._actions = ACTIONS[action_size].values()
        if display:
            width = 640
            height = 480

        self.env = deepmind_lab.Lab(
            level, ['RGB_INTERLACED'],
            config={
                'fps': str(fps),
                'width': str(width),
                'height': str(height)
            })

        self.s_t = None
        self.reset()

    def state(self):
        return self.s_t

    def act(self, action):
        reward, terminal, self.s_t = self._process_frame(self._actions[action])
        return reward, terminal

    def _process_frame(self, action):
        reward = self.env.step(action, num_steps=self._frame_skip)
        terminal = not self.env.is_running()

        if terminal:
            return reward, terminal, None

        # train screen shape is (84, 84) by default
        screen = self.env.observations()['RGB_INTERLACED']
        x_t = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
        if self._display:
            x_t = imresize(x_t, (self._width, self._height))

        x_t = x_t.astype(np.float32)
        x_t *= (1.0 / 255.0)
        return reward, terminal, x_t

    def reset(self):
        while True:
            self.env.reset()

            # randomize initial state
            if self._no_op_max > 0:
                no_op = np.random.randint(0, self._no_op_max + 1)
                for _ in range(no_op):
                    action = random.choice(self._actions)
                    self.env.step(action, num_steps=self._frame_skip)

            _, terminal, self.s_t = self._process_frame(random.choice(self._actions))
            if not terminal:
                break


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

ACTIONS = {'f': FULL_ACTIONS,
           'full': FULL_ACTIONS,
           'b': FULL_ACTIONS,
           'big': FULL_ACTIONS,
           'm': MEDIUM_ACTIONS,
           'medium': MEDIUM_ACTIONS,
           's': SMALL_ACTIONS,
           'small': SMALL_ACTIONS}
