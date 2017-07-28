from builtins import object

class Batch(object):
    def __init__(self, parameter_server, exploit):
        self.exploit = exploit
        self.ps = parameter_server
        self.reset()
        self.last_state = None
        self.last_action = None

    @property
    def experience(self):
        return self.episode.experience

    def begin(self):
        self.episode.begin()

    def step(self, reward, state, terminal):
        return None

    def end(self):
        experience = self.episode.end()
        if not self.exploit:
            self.apply_gradients(self.compute_gradients(experience), len(experience))
            self.send_experience(experience)

    def reset(self):
        pass

    # Helper methods
    def send_experience(self, experience):
        pass
