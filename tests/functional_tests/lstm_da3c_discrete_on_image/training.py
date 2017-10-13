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
        self.env = MazeEnv(env='level_2')

    def episode(self, number):
        n_episode = 1
        while True:
            episode_reward = 0
            state = self.env.reset()
            action = self.agent.update(reward=None, state=state)
            terminal = False
            while not terminal:
                state, reward, terminal, info = self.env.step(action)
                episode_reward += reward
                action = self.agent.update(reward=reward, state=state, terminal=terminal)
            print('Episode %d, reward %f' % (n_episode, episode_reward))
            n_episode += 1


class MazeEnv(object):
    moves = [np.array([-1, 0]),  # go up
             np.array([0, -1]),  # go left
             np.array([1, 0]),   # go down
             np.array([0, 1])]   # go right
    actions = ['^', '<', 'v', '>']
    step_cost = .01
    goal_cost = 1.0
    max_steps = 100

    def __init__(self, env='level_1', shape=(7, 7), no_op_max=0):
        self.env = env
        self._level = self._read_level(env)
        self._no_op_max = no_op_max

        self.shape = list(shape + (3,))
        assert len(self.shape) == 3, "You should provide shape as [x, y]"
        self.range = [int((self.shape[0]-1)/2), int((self.shape[1]-1)/2)]

        self.action_size = len(MazeEnv.actions)
        self.timestep_limit = MazeEnv.max_steps

        self._maze, self._step_count = None, None
        self._goal_pos, self._player_pos = None, None
        self._episode_reward = 0

        if env[-1] == '1':
            self._init_maze = self._init_random_maze
        else:
            self._spawns, self._goal = self._read_spawns(env)
            self._init_maze = self._init_spawn_maze
        self.reset()

    def step(self, action):
        reward, terminal = self._step(action)
        state = self._process_state()
        self._episode_reward += reward

        if terminal:
            info = {'episode_reward': self._episode_reward}
        else:
            info = {}
        return state, reward, terminal, info

    def reset(self):
        while True:
            self._init_maze()
            terminal = False

            if self._no_op_max != 0:
                no_op = np.random.randint(0, self._no_op_max)
                for _ in range(no_op):
                    reward, terminal = self._step(np.random.randint(0, self.action_size))

            if not terminal:
                state = self._process_state()
                break

        self._step_count = 0
        self._episode_reward = 0
        return state

    def _step(self, action):
        new_pos = self._player_pos + MazeEnv.moves[action]

        if new_pos[0] == self._goal_pos[0] and new_pos[1] == self._goal_pos[1]:
            return MazeEnv.goal_cost, True
        else:
            self._step_count += 1
            if self._maze[new_pos[0], new_pos[1], 0] != 1:
                self._maze[self._player_pos[0], self._player_pos[1], 2] = 0
                self._player_pos = new_pos
                self._maze[self._player_pos[0], self._player_pos[1], 2] = 1
            return -MazeEnv.step_cost, self._step_count >= MazeEnv.max_steps

    def _process_state(self):
        region = self._maze[self._player_pos[0]-self.range[0]:1+self._player_pos[0]+self.range[0],
                            self._player_pos[1]-self.range[1]:1+self._player_pos[1]+self.range[1],
                            ...]
        state = np.copy(region)
        return state

    @staticmethod
    def _read_level(level_name):
        lvl_read = []
        with open('maps/' + level_name + '.txt', 'r') as lvl_file:
            for i, line in enumerate(lvl_file):
                lvl_read.append([])
                for j in line[:-1]:
                    if j in ('0', '1'):
                        lvl_read[i].append(int(j))
                    else:
                        logger.error("You map file should be defined with '0' and '1'!")
                        sys.exit(-1)
        return np.asarray(lvl_read)

    def _init_random_maze(self):
        # init goal position
        goal = np.zeros_like(self._level)
        while True:
            row_idx = np.random.randint(0, self._level.shape[0])
            col_idx = np.random.randint(0, self._level.shape[1])
            if self._level[row_idx, col_idx] == 0:
                goal[row_idx, col_idx] = 1
                self._goal_pos = np.array([row_idx, col_idx])
                break

        # init player position
        player = np.zeros_like(self._level)
        while True:
            row_idx = np.random.randint(0, self._level.shape[0])
            col_idx = np.random.randint(0, self._level.shape[1])
            if self._level[row_idx, col_idx] == 0 and goal[row_idx, col_idx] == 0:
                player[row_idx, col_idx] = 1
                self._player_pos = np.array([row_idx, col_idx])
                break

        # stack all together in depth (along third axis)
        self._maze = np.dstack((self._level, goal, player))

    def _init_spawn_maze(self):
        # init player position
        player = np.zeros_like(self._level)

        idx = np.random.randint(0, self._spawns.shape[0])
        self._player_pos = self._spawns[idx]
        player[self._player_pos[0], self._player_pos[1]] = 1

        # stack all together in depth (along third axis)
        self._maze = np.dstack((self._level, self._goal, player))

    def _read_spawns(self, env):
        spawn_read = []
        with open('maps/spawn_' + env[-1] + '.txt', 'r') as spawn_file:
            for line in spawn_file:
                row, col = line.split()
                spawn_read.append([int(row), int(col)])

        self._goal_pos = np.array(spawn_read[-1])
        del spawn_read[-1]

        goal = np.zeros_like(self._level)
        goal[self._goal_pos[0], self._goal_pos[1]] = 1

        return np.asarray(spawn_read), goal


if __name__ == '__main__':
    Training().run()
