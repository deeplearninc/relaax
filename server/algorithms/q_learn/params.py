class Params(object):
    def __init__(self):
        self.default_params = {'grid_size': 64,
                               'action_size': 4,
                               'batch_size': 40,
                               'gamma': 0.975}
        self.grid_size = None
        self.action_size = None
        self.batch_size = None
        self.gamma = None
