import argparse


class Params(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--scope", type=str, default="ale_model", help="Name of model scope")
        parser.add_argument("--algo", type=str, default="a3c", help="Name of the RL algorithm to perform")
        parser.add_argument("--env", type=str, default="Boxing-v0", help="Name of the Environment")
        parser.add_argument("--agents", type=int, default=8, help="Number of parallel training Agents")
        parser.add_argument("--lstm", type=bool, default=True, help="Adds LSTM layer before net output")

        self.args = parser.parse_args()
        if self.args.algo != "a3c":
            self.args.agents = 1

        self.game_rom = self.args.env           # Name of the Atari game ROM
        self.threads_cnt = self.args.agents     # Number of parallel training agents

        self.action_type = True        # Type of returning action (True == Discrete, False == float Box)
        self.action_size = None         # Assign later at Game(s) launching
        self.screen_width = 84          # Screen width
        self.screen_height = 84         # Screen height
        self.history_len = 4            # Number of most recent frames experiences by the agent
        self.episode_len = 5            # local loop size for one episode

        self.use_GPU = False            # to use GPU, set to the True
        self.use_LSTM = self.args.lstm  # to use LSTM instead of FF, set to the True
        # self.max_global_step = None     # amount of maximum global steps to pass through the training
