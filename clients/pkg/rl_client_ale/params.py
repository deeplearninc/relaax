class Params(object):
    def __init__(self, args):
        self.args = args
        if self.args.algo != "a3c":
            self.args.agents = 1

        self.game_rom = self.args.game
        self.threads_cnt = self.args.agents

        self.action_size = None     # Assign later at Game(s) launching
        self.screen_width = 84      # Screen width
        self.screen_height = 84     # Screen height
        self.history_len = 4        # Number of most recent frames experiences by the agent
        self.episode_len = 5        # local loop size for one episode

        self.use_GPU = False            # to use GPU, set to the True
        self.use_LSTM = self.args.lstm  # to use LSTM instead of FF, set to the True
        # self.max_global_step = None     # amount of maximum global steps to pass through the training
