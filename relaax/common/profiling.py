from __future__ import absolute_import
from builtins import range
from builtins import object

import json
import os
import sys
import threading
import time


def get_profiler(name):
    return Profiler.get_profiler(name)


# TODO: enable/disable for profiler hierarhcy (move it to root profiler)
def enable(enabled):
    Profiler.enabled = enabled


def set_handlers(handlers):
    Profiler.handlers = handlers

def get_handlers():
    return Profiler.handlers

def add_handler(handler):
    if handler not in Profiler.handlers:
        Profiler.handlers.append(handler)

def remove_handler(handler):
    if handler in Profiler.handlers:
        Profiler.handlers.remove(handler)

def wrap(f):
    return Profiler.root.wrap(f)


class Handler(object):
    def emit(self, record):
        raise NotImplementedError()


class StreamHandler(Handler):
    def __init__(self, stream=sys.stderr, prefix='PROF', suffix='FORP'):
        super(StreamHandler, self).__init__()
        self.stream = stream
        self.pattern = '%s%%s%s\n' % (prefix, suffix)

    def emit(self, record):
        self.stream.write(self.pattern % json.dumps(record))


class Profiler(object):
    # TODO: introduce enable/disable context manager
    enabled = False
    root = None
    profilers = {}
    handlers = [StreamHandler()]

    def __init__(self, name):
        self.name = name

    def wrap(self, f):
        event = CompleteEvent(self, f.__qualname__)
        def wrapper(*args, **kwargs):
            with event:
                return f(*args, **kwargs)
        return wrapper

    def handle(self, record):
        for h in self.handlers:
            h.emit(record)

    @classmethod
    def get_profiler(cls, name):
        if name not in cls.profilers:
            cls.profilers[name] = Profiler(name)
        return cls.profilers[name]


Profiler.root = Profiler('')


class BaseEvent(object):
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def handle(self, **kwargs):
        kwargs['cat'] = self.profiler.name
        kwargs['name'] = self.name
        kwargs['pid'] = os.getpid()
        kwargs['tid'] = threading.current_thread().ident
        self.profiler.handle(kwargs)


class CompleteEvent(BaseEvent):
    def __init__(self, profiler, name):
        super(CompleteEvent, self).__init__(profiler, name)
        self.start = None

    def __enter__(self):
        if self.profiler.enabled:
            self.start = int(1000000 * time.time())

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiler.enabled:
            finish = int(1000000 * time.time())
            self.handle(ph='X', ts=self.start, dur=finish - self.start)
