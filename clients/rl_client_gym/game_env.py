import gym
import numpy as np
from scipy.misc import imresize


class Env:
    def __init__(self, params, seed=0, display=False, no_op_max=7):
        self.display = display
        self.gym = gym.make(params.game_rom)
        self.gym.seed(seed)
        self.dims = (params.screen_height, params.screen_width)

        self._no_op_max = no_op_max
        self.state = np.empty((210, 160, 1), dtype=np.uint8)
        self.terminal = False
        self.reset()

    def getActions(self):
        return self.gym.action_space.n

    def reset(self):
        self.gym.reset()
        no_op = np.random.randint(0, self._no_op_max)
        for _ in range(no_op):
            self.gym.step(0)

        self.state = self.getState(self.gym.step(0)[0], False)
        self.terminal = False

    @staticmethod
    def getState(screen, reshape=True):
        gray = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
        gray = gray.astype(np.uint8)

        resized_screen = imresize(gray, (110, 84))
        state = resized_screen[18:102, :]

        if reshape:
            state = np.reshape(state, (84, 84, 1))

        return state

    def act(self, action):
        if self.display:
            self.gym.render()
        screen, reward, self.terminal, info = self.gym.step(action)
        self.state = self.getState(screen)
        if self.terminal:
            self.gym.reset()
        return reward
