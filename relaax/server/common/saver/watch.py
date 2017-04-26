class Watch(object):
    def __init__(self, ps, *intervals):
        self.ps = ps
        self.interval = Intervals(intervals)
        self.last_saved_step = ps.n_step()

    def check(self):
        if self.interval.check():
            n_step = self.ps.n_step()
            if n_step != self.last_saved_step:
                self.ps.save_checkpoint()
                self.last_saved_step = n_step
            self.interval.reset()


class Intervals(object):
    def __init__(self, intervals):
        self.intervals = []
        for interval, counter in intervals:
            if interval is not None:
                self.intervals.append(Interval(interval, counter))

    def check(self):
        for i in self.intervals:
            if i.check():
                return True
        return False

    def reset(self):
        for i in self.intervals:
            i.reset()


class Interval(object):
    def __init__(self, interval, counter):
        self.interval = interval
        self.counter = counter
        self.reset()

    def check(self):
        return self.target <= self.counter()

    def reset(self):
        self.target = self.counter() + self.interval
