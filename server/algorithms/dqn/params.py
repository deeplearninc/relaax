class Params(object):
    def __init__(self):
        self.default_params = {'game_rom': None,
                               'action_size': None,
                               'screen_width': 84,
                               'screen_height': 84,
                               'history_len': 4,
                               'episode_len': 4,    # =5 on the Client by A3C prerequisites
                               'use_GPU': False,
                               'max_global_step': 10 ** 8}
        self.game_rom = None        # Name of the given game rom
        self.action_size = None     # Action size for given game rom
        self.screen_width = None    # Screen width
        self.screen_height = None   # Screen height
        self.history_len = None     # Number of most recent frames experiences by the agent

        self.episode_len = None  # local loop size for one episode
        self.use_GPU = None      # to use GPU, set to the True
        self.max_global_step = None  # amount of maximum global steps to pass through the training

        # Algorithm's predefined parameters, which isn't sent to the Client
        self.memory_size = 10 ** 5  # Memory size for deque from Memory class in memory.py

        # DQN parameters
        self.lr = 0.00025           # Learning rate
        self.lr_anneal = 20000      # Step size of learning rate annealing
        self.discount = 0.99        # Discount rate
        self.batch_size = 32        # Batch size for Memory class
        self.accumulator = 'mean'   # Batch accumulator
        self.decay_rate = 0.95      # Decay rate for RMSProp
        self.min_decay_rate = 0.01  # Min decay rate for RMSProp
        self.init_eps = 1.0         # Initial value of e in e-greedy exploration
        self.final_eps = 0.1        # Final value of e in e-greedy exploration
        self.final_eps_frame = 10 ** 6  # The number of frames over which the
        # initial value of e is linearly annealed to its final
        self.clip_delta = 1.0       # Clip error term in update between this number and its negative
        self.steps = 10 ** 4        # Copy main network to target network after this many steps
        self.replay_start_size = 50000  # A uniform random policy is run for this number of frames
        #  before learning starts and the resulting experience is used to populate the replay memory
        self.save_weights = 2 * 10**6   # Save the model after this amount of steps

        # Miscellaneous parameters
        self.episodes = 100         # Number of episodes
        self.random_starts = 30     # Perform max this number of no-op actions
        #  to be performed by the agent at the start of an episode
        self.start_from = 0         # global step from which training should continue after restoring
