import argparse


class Params(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--algo", type=str, default="a3c", help="Algorithm is selected to Train")
        parser.add_argument("--game", type=str, default="Boxing-v0", help="Name of the Atari game ROM")
        parser.add_argument("--agents", type=int, default=8, help="Number of parallel training Agents")
        args = parser.parse_args()

        if args.algo != "a3c":
            args.agents = 1

        self.algo = args.algo       # Name of the selected algorithm to train agent(s)
        self.game_rom = args.game   # Name of the Atari game ROM
        self.action_size = None     # Assign later at Game(s) launching
        self.screen_height = 84     # Screen height
        self.screen_width = 84      # Screen width
        self.history_len = 4        # Number of most recent frames experiences by the agent

        self.threads_cnt = args.agents  # number of parallel training agents
        self.episode_len = 5            # local loop size for one episode
        self.use_GPU = False            # to use GPU, set to the True
        self.use_LSTM = True            # to use LSTM instead of FF, set to the True
        # self.max_global_step = None     # amount of maximum global steps to pass through the training
