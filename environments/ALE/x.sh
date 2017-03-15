#!/bin/bash

$3 &
./main --rlx-server $1 --rom $2 --display true &
x11vnc --usepw --forever
