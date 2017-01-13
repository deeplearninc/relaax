#!/bin/bash

$3 &
./main --rlx-server $1 --env $2 &
x11vnc --usepw --forever