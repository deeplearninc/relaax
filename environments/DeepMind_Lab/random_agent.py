#!/usr/bin/env python
import relaax
import sys
import os

path_to_lab_client = \
    str(relaax.__path__[0][:-6]) + 'environments/DeepMind_Lab/main'
args = [path_to_lab_client] + sys.argv[1:]
os.execv(path_to_lab_client, args)
