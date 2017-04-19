import experience


class Episode(object):
    def __init__(self, *args):
        self.keys = args
        self.experience = None

    def begin(self):
        assert self.experience is None
        self.experience = experience.Experience(*self.keys)

    def step(self, **kwargs):
        assert self.experience is not None
        self.experience.push_record(**kwargs)

    def end(self):
        assert self.experience is not None
        experience = self.experience
        self.experience = None
        return experience
