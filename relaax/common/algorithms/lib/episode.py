import experience


class Episode(object):
    def __init__(self, *args):
        self.keys = args
        self.in_episode = False
        self.experience = {k: [] for k in self.keys}
        self.experience_size = 0
        # 0: number of experience lists of self.experience_size     length
        # 1: number of experience lists of self.experience_size + 1 length
        # 2: number of experience lists of self.experience_size + 2 length
        self.counts = [len(args), 0, 0]

    def begin(self):
        assert not self.in_episode
        self.in_episode = True

    def step(self, **kwargs):
        assert self.in_episode

        for k, v in kwargs.iteritems():
            if v is not None:
                self.push_value(self.experience[k], v)

        assert self.counts[0] >= 0
        if self.counts[0] == 0:
            self.inc_size()

    def end(self):
        assert self.in_episode

        experience = self.experience

        # remove all complete experience records from experience
        self.experience = {k: experience[k][self.experience_size:] for k in self.keys}

        # delete incomplete experience from experience to return
        for k, v in experience.iteritems():
            if len(v) > self.experience_size:
                del v[-1]
                assert len(v) == self.experience_size

        self.experience_size = 0

        self.in_episode = False
        return experience

    def inc_size(self):
        self.experience_size += 1
        self.counts[0] = self.counts[1]
        self.counts[1] = self.counts[2]
        self.counts[2] = 0

    def push_value(self, experience_list, value):
        self.counts[len(experience_list) - self.experience_size] -= 1
        experience_list.append(value)
        self.counts[len(experience_list) - self.experience_size] += 1
