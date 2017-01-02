import deepmind_lab
import numpy as np
import random


class GameProcessFactory(object):
    def __init__(self, level):
        self._level = level

    def new_env(self, seed):
        return _GameProcess(seed, self._level)

    def new_display_env(self, seed):
        return _GameProcess(seed, self._level, display=True, no_op_max=0)


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class _GameProcess(object):
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
    ACTION_LIST = ACTIONS.values()

    def __init__(self, fps, level, display=False, frame_skip=4, no_op_max=7):
        self._frame_skip = frame_skip
        self._no_op_max = no_op_max
        width = 84
        height = 84

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
        reward, terminal, self.s_t = self._process_frame(_GameProcess.ACTION_LIST[action])
        return reward, terminal

    def _process_frame(self, action):
        reward = self.env.step(action, num_steps=self._frame_skip)
        terminal = not self.env.is_running()

        if terminal:
            return reward, terminal, None

        # screen shape is (84, 84)
        screen = self.env.observations()['RGB_INTERLACED']
        x_t = np.dot(screen[..., :3], [0.299, 0.587, 0.114])

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
                    action = random.choice(_GameProcess.ACTION_LIST)
                    self.env.step(action, num_steps=self._frame_skip)

            _, terminal, self.s_t = self._process_frame(random.choice(_GameProcess.ACTION_LIST))
            if not terminal:
                break
