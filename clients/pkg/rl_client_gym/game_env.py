import gym
import numpy as np
from scipy.misc import imresize


AtariList = ['AirRaid-v0', 'Alien-v0', 'Amidar-v0', 'Assault-v0', 'Asterix-v0',
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


class StateDeterminator(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class EnvFactory(object):
    def __init__(self, game_rom):
        self._game_rom = game_rom

    def new_env(self, seed):
        return Env(self._game_rom, seed)

    def new_display_env(self, seed):
        return Env(self._game_rom, seed, display=True)


class Env(object):
    def __init__(self, game_rom, seed, display=False, no_op_max=7):
        self.display = display
        self.gym = gym.make(game_rom)
        self.gym.seed(seed)
        self._no_op_max = no_op_max

        self._state = None   # np.empty((210, 160, 1), dtype=np.uint8)

        self.getState = StateDeterminator(self.getStateAll)
        if game_rom in AtariList:
            self.getState = StateDeterminator(self.getStateAtari)
        self.reset()

    def action_size(self):
        return self.gym.action_space.n

    def state(self):
        return self._state

    def act(self, action):
        if self.display:
            self.gym.render()
        screen, reward, terminal, info = self.gym.step(action)
        self._state = self.getState(screen)
        return reward, terminal

    def getActionSpace(self):
        return self.gym.action_space

    def getStateSpace(self):
        return self.gym.observation_space

    def reset(self):
        while True:
            self.gym.reset()
            terminal = False
            no_op = np.random.randint(0, self._no_op_max)
            for _ in range(no_op):
                screen, _, terminal, _ = self.gym.step(0)

            if not terminal:
                self._state = self.getState(screen, False)
                break

    @staticmethod
    def getStateAtari(screen, reshape=True):
        gray = np.dot(screen[..., :3], [0.299, 0.587, 0.114])
        # gray = gray.astype(np.uint8)

        resized_screen = imresize(gray, (110, 84))
        state = resized_screen[18:102, :]

        if reshape:
            state = np.reshape(state, (84, 84, 1))
        state = state.astype(np.float32)
        state *= (1.0 / 255.0)
        return state

    @staticmethod
    def getStateAll(state, reshape=True):
        if reshape:
            state = np.reshape(state, (state.shape[0], state.shape[1], 1))
        return state
