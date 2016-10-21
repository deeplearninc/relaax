class Params(object):
    def __init__(self, args):
        self.args = args
        self.game_rom = args.game
        self.threads_cnt = args.agents
