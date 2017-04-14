import experience


class Episode(object):
    def __init__(self):
        self.values = None
        self.experience = None

    @property
    def in_episode(self):
        return self.experience is not None

    @property
    def length(self):
        assert self.in_episode
        return self.experience.length

    def begin(self):
        assert not self.in_episode
        self.values = [None] * 3
        self.experience = experience.Experience()

    def step(self, reward, state, action):
        assert self.in_episode

        values = [state, action, reward]
        for i in xrange(len(self.values)):
            if values[i] is not None and self.values[i] is None:
                self.values[i] = values[i]
                values[i] = None

        if all(v is not None for v in self.values):
            self.experience.accumulate(*self.values)
            self.values = values
        else:
            assert all(v is None for v in values)

    def end(self):
        assert self.in_episode
        assert all(v is None for v in self.values)
        experience = self.experience
        self.values = None
        self.experience = None
        return experience
