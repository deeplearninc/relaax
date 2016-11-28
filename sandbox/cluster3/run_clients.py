from __future__ import print_function

import argparse
import os
import re
import signal
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default=None, help='agent server address (host:port)')
    parser.add_argument("--game", type=str, default="boxing", help="Name of the Atari game ROM")
    return parser.parse_args()


class Handler(object):
    def __init__(self, game, agent):
        self._game = game
        self._agent = agent
        self._n_clients = 0
        self._clients = []

    def watch(self):
        stop = [False]

        def handler(_1, _2):
            stop[0] = True
            print('')

        print('Press Ctrl+C to stop watching...')

        old_handler = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, handler)
            last_n_clients = None
            while not stop[0]:
                n_clients = self._calc_n_clients()
                if n_clients != last_n_clients:
                    print('clients: %d' % n_clients)
                if n_clients < self._n_clients:
                    self._run_clients(self._n_clients - n_clients)
                if self._n_clients < n_clients:
                    self._kill_clients(n_clients - self._n_clients)
                last_n_clients = n_clients
                time.sleep(1)
        finally:
            signal.signal(signal.SIGINT, old_handler)

    def start(self, n):
        self._n_clients += n

    def stop(self, n):
        self._n_clients -= n
        if self._n_clients < 0:
            self._n_clients = 0

    def tail(self, i):
        print('Press Ctrl+C to stop watching...')
        os.system('tail -n40 -F out/client_%d' % i)

    def _calc_n_clients(self):
        c = 0
        for i in xrange(len(self._clients)):
            if self._clients[i].poll() is None:
                c += 1
        return c

    def _run_clients(self, n):
        c = 0
        for i in xrange(len(self._clients)):
            if self._clients[i].poll() is not None:
                if c < n:
                    print('Starting client %d...' % i)
                    self._clients[i] = subprocess.Popen(
                        ['bash', 'run_client.sh', self._agent, str(i)],
                        preexec_fn=os.setpgrp
                    )
                    c += 1
        for i in xrange(len(self._clients), len(self._clients) + n - c):
            print('Starting client %d...' % i)
            self._clients.append(subprocess.Popen(
                ['bash', 'run_client.sh', self._agent, str(i)],
                preexec_fn=os.setpgrp
            ))

    def _kill_clients(self, n):
        c = 0
        for i in reversed(xrange(len(self._clients))):
            if c < n:
                if self._clients[i].poll() is None:
                    print('Stopping client %d...' % i)
                    self._clients[i].send_signal(signal.SIGINT)
                    c += 1


def arg(s, default):
    if s != '':
        return int(s)
    return default


def help():
    print('')
    print('This is tool to start and stop clients.')
    print('')
    print('The commands are:')
    print('?  - show this help;')
    print('w  - watch for number of working clients;')
    print('N+ - start N clients;')
    print('N- - stop N clients;')
    print('Nt - tail Nth client output.')
    print('')

def loop(handler):
    while True:
        s = raw_input('(? - help) > ')
        if s == '?':
            help()
        elif s == 'w':
            handler.watch()
        elif s == 'q':
            break
        else:
            m = re.match('^(\d*)([-\+t])$', s)
            if m is not None:
                s = m.group(1)
                if m.group(2) == '+':
                    arg_ = arg(m.group(1), 1)
                    if arg_ == 1:
                        print('Starting new client...')
                    else:
                        print('Starting %d new clients...' % arg_)
                    handler.start(arg_)
                    handler.watch()
                elif m.group(2) == '-':
                    arg_ = arg(m.group(1), 1)
                    if arg_ == 1:
                        print('Stopping client...')
                    else:
                        print('Stopping %d clients...' % arg_)
                    handler.stop(arg_)
                    handler.watch()
                elif m.group(2) == 't':
                    handler.tail(arg(m.group(1), 0))
                    print('')


def main():
    args = parse_args()
    help()
    loop(Handler(args.game, args.agent))


if __name__ == '__main__':
    main()
