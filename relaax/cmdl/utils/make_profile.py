#!/usr/bin/env python

import argparse
import json
import logging


parser = argparse.ArgumentParser(description=(
                                 'Converts text profile data into JSON format compatible with Chrome.'))
parser.add_argument('profile', type=str, nargs='+', help='profile files')

args = parser.parse_args()

records = []
for fname in args.profile:
    with open(fname, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
            except ValueError as e:
                logging.exception(e)
            else:
                records.append(record)

print(json.dumps({'traceEvents': records}))
