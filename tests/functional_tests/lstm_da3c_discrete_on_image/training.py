from __future__ import print_function
from builtins import range

import logging
import math
import numpy as np
from PIL import Image, ImageDraw
import random


from relaax.environment.config import options
from relaax.environment.training import TrainingBase
from relaax.common.rlx_message import RLXMessageImage


logger = logging.getLogger(__name__)


class Training(TrainingBase):
    def __init__(self):
        super(Training, self).__init__()
        self.env = CustomEnv()

    def episode(self, number):
        while True:
            episode_size = 200
            episode_reward = 0
            state = self.env.reset()
            action = self.agent.update(reward=None, state=state)
            for step in range(episode_size):
                # old_speed = self.env.speed
                state, reward, terminal, info = self.env.step(action)
                episode_reward += reward
                if step == episode_size - 1:
                    terminal = True
                # print('old speed %s, new speed %s, reward %f' % (old_speed, self.env.speed, reward))
                action = self.agent.update(reward=reward, state=state, terminal=terminal)
            print('episode reward is %f' % episode_reward)


class CustomEnv(object):
    def __init__(self):
        self.place = 0
        self.speed = 4
        self.ball = Ball()

    def reset(self):
        return self.state()

    # state, reward, terminal, info = env.step(action.argmax())
    def step(self, action):
        old_speed = self.speed
        assert 0 <= action <= 2
        # self.speed = max(1, min(self.speed + (action - 1), 7))
        # reward = self.reward()
        if self.speed <= 4:
            reward = 1 if action == 1 else 0
        else:
            reward = 1 if action == 0 else 0

        #print('old speed %s, new speed %s, reward %f' % (old_speed, self.speed, reward))
        self.speed = max(1, min(self.speed + random.randint(-1, 1), 7))
        assert 1 <= self.speed <= 7
        self.place += self.speed
        state = self.state()
        return state, reward, False, None

    def state(self):
        return self.ball.paint(self.place * math.pi / 29)

    def reward(self):
        if self.speed in [1, 7]:
            return 0
        if self.speed in [2, 3, 5, 6]:
            return 0.5
        return 1

    def render(self):
        pass


class Ball(object):
    SIZE = 84
    R1 = 30
    R2 = 5

    def paint(self, alpha):
        image = Image.new('L', (self.SIZE, self.SIZE))
        draw = ImageDraw.Draw(image)
        # fill image background with black
        draw.rectangle((0, 0, self.SIZE, self.SIZE), fill = 'black', outline ='black')
        x0 = self.SIZE / 2
        y0 = self.SIZE / 2
        x = x0 + self.R1 * math.cos(alpha)
        y = y0 - self.R1 * math.sin(alpha)
        # draw.ellipse((x0 - self.R2, y0 - self.R2, x0 + self.R2, y0 + self.R2), fill = 'gray', outline ='gray')
        # draw.ellipse((x - self.R2, y - self.R2, x + self.R2, y + self.R2), fill = 'white', outline ='white')
        draw.line(((x0, y0), (x, y)), fill='white', width=self.R2)
        return RLXMessageImage(image)


if __name__ == '__main__':
    Training().run()
