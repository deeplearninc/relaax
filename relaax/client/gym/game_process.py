import numpy as np
from scipy.misc import imresize

import gym    # you should install gym via pip
from gym.spaces import Box  # check continuous


class GameProcessFactory(object):
    def __init__(self, env):
        self._env = env

    def new_env(self, seed):
        return _GameProcess(seed, self._env)

    def new_display_env(self, seed):
        return _GameProcess(seed, self._env, display=True, no_op_max=0)


class SetProcessFunc(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class _GameProcess(object):
    AtariGameList = [
        'AirRaid-v0', 'Alien-v0', 'Amidar-v0', 'Assault-v0', 'Asterix-v0',
        'Asteroids-v0', 'Atlantis-v0', 'BankHeist-v0', 'BattleZone-v0', 'BeamRider-v0',
        'Berzerk-v0', 'Bowling-v0', 'Boxing-v0', 'Breakout-v0', 'Carnival-v0',
        'Centipede-v0', 'ChopperCommand-v0', 'CrazyClimber-v0', 'DemonAttack-v0', 'DoubleDunk-v0',
        'ElevatorAction-v0', 'Enduro-v0', 'FishingDerby-v0', 'Freeway-v0', 'Frostbite-v0',
        'Gopher-v0', 'Gravitar-v0', 'IceHockey-v0', 'Jamesbond-v0', 'JourneyEscape-v0',
        'Kangaroo-v0', 'Krull-v0', 'KungFuMaster-v0', 'MontezumaRevenge-v0', 'MsPacman-v0',
        'NameThisGame-v0', 'Phoenix-v0', 'Pitfall-v0', 'Pong-v0', 'Pooyan-v0',
        'PrivateEye-v0', 'Qbert-v0', 'Riverraid-v0', 'RoadRunner-v0', 'Robotank-v0',
        'Seaquest-v0', 'Skiing-v0', 'Solaris-v0', 'SpaceInvaders-v0', 'StarGunner-v0',
        'Tennis-v0', 'TimePilot-v0', 'Tutankham-v0', 'UpNDown-v0', 'Venture-v0',
        'VideoPinball-v0', 'WizardOfWor-v0', 'YarsRevenge-v0', 'Zaxxon-v0']

    def __init__(self, rand_seed, env, display=False, no_op_max=7):
        self.gym = gym.make(env)
        self.gym.seed(rand_seed)
        self._no_op_max = no_op_max

        self.display = display
        self._close_display = False

        self.timestep_limit = self.gym.spec.timestep_limit
        self.cur_step_limit = None
        self._state = None

        self._process_state = SetProcessFunc(self._process_atari)
        if env in _GameProcess.AtariGameList:
            self._process_state = SetProcessFunc(self._process_all)

        self.ac_size, self.box = self._action_size()
        self.reset()

    def _action_size(self):
        space = self.gym.action_space
        if isinstance(space, Box):
            return space.shape[0], True
        return space.n, False

    def state(self):
        return self._state

    def act(self, action):
        if self.box:
            action = np.clip(action, self.gym.action_space.low, self.gym.action_space.high)
        if self.display:
            if self._close_display:
                self.gym.render(close=True)
                self.display = False
                self._close_display = False
            else:
                self.gym.render()

        state, reward, terminal, info = self.gym.step(action)
        self._state = self._process_state(state)

        self.cur_step_limit += 1
        if self.cur_step_limit > self.timestep_limit:
            terminal = True

        return reward, terminal

    def reset(self):
        do_act = 0
        self.cur_step_limit = 0

        while True:
            self.gym.reset()
            if not self.display:
                no_op = np.random.randint(0, self._no_op_max)
                for _ in range(no_op):
                    if self.box:
                        do_act = self._safe_rnd_act()
                    self.gym.step(do_act)

            if self.box:
                do_act = self._safe_rnd_act()
            env_state = self.gym.step(do_act)

            self._state = self._process_state(env_state[0])
            if not env_state[2]:
                break

    @staticmethod
    def _process_atari(screen):
        gray = np.dot(screen[..., :3], [0.299, 0.587, 0.114])

        resized_screen = imresize(gray, (110, 84))
        state = resized_screen[18:102, :]

        state = state.astype(np.float32)
        state *= (1.0 / 255.0)
        return state

    @staticmethod
    def _process_all(state):
        return state

    def _safe_rnd_act(self):
        act = (np.random.randn(1, self.ac_size).astype(np.float32))[0]
        return np.clip(act, self.gym.action_space.low, self.gym.action_space.high)
