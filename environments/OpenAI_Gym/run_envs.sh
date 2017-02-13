#!/bin/bash

# This script allows to run multiply gym's environments via CLI (without dockers).
# Just run this script as follows: ./run_envs.sh number_of_environments

N=$1                # number of environments to run (1st positional argument)
ENV_NAME=$2         # name of the gym's environment to run (2nd positional argument)
RLX=localhost:7001  # address of RLX server [IP:PORT]
VIRTUAL_ENV=tst     # name of anaconda virtual environment (leave blank if you don't use it)

LOG_FOLDER="/tmp/out"
mkdir -p -- "$LOG_FOLDER"  # log folder to which clients write its output


if [ -z "$ENV_NAME" ]
then
    ENV_NAME="BipedalWalker-v2"
fi

if [ -z "$VIRTUAL_ENV" ]
then
    for i in `seq 0 $((N - 1))`;
    do
        echo running client $i
        ./main --rlx-server "$RLX" --env "$ENV_NAME" &>"$LOG_FOLDER"/client_$i &
        sleep 1
    done
else
    for i in `seq 0 $((N - 1))`;
    do
        echo running client $i
        source activate "$VIRTUAL_ENV" && ./main --rlx-server "$RLX" --env "$ENV_NAME" &>"$LOG_FOLDER"/client_$i &
        sleep 1
    done
fi

read -p "Press [Enter] to terminate all running clients..."

kill $(ps ax | grep "$ENV_NAME")

sleep 2

ps ax | grep "$ENV_NAME"