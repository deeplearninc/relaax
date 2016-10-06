
class Environment(object):
    def action_size(self):
        raise NotImplementedError()

    def state(self):
        raise NotImplementedError()

    def act(self, action):
        raise NotImplementedError()
