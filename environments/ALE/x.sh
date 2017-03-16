#!/bin/bash

# DEFAULT VALUES
NUM=1    # number of clients to run
ARGS=""  # concat all args to one string
path_to_roms="/roms"  # default path to the game roms inside the docker (needs to know for volume mounting)

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -x|--rlx-server)
    shift # rm argument
    RLX="$1"
    ARGS+="--rlx-server $RLX "
    shift # rm value
    ;;
    -x=*|--rlx-server=*)
    RLX="${key#*=}"
    ARGS+="--rlx-server $RLX "
    shift # rm argument=value
    ;;
    -e|--env)
    shift # rm argument
    ENV="$1"
    ARGS+="--rom ${path_to_roms}/${ENV}.bin "
    shift # rm value
    ;;
    -e=*|--env=*)
    ENV="${key#*=}"
    ARGS+="--rom ${path_to_roms}/${ENV}.bin "
    shift # rm argument=value
    ;;
    -s|--seed)
    shift # rm argument
    SEED="$1"
    ARGS+="--seed $SEED "
    shift # rm value
    ;;
    -s=*|--seed=*)
    SEED="${key#*=}"
    ARGS+="--seed $SEED "
    shift # rm argument=value
    ;;
    -f|--frame-skip)
    shift # rm argument
    SKIP="$1"
    ARGS+="--frame-skip $SKIP "
    shift # rm value
    ;;
    -f=*|--frame-skip=*)
    SKIP="${key#*=}"
    ARGS+="--frame-skip $SKIP "
    shift # rm argument=value
    ;;
    -n|--num)
    shift # rm argument
    NUM="$1"
    shift # rm value
    ;;
    -n=*|--num=*)
    NUM="${key#*=}"
    shift # rm argument=value
    ;;
    -d|--display)
    VISUAL_OUTPUT="YES"
    ARGS+="--display true "
    shift # rm argument
    ;;
    *) # unknown option
    shift
    ;;
esac
done

echo "ARGS = ${ARGS}"

for i in `seq 0 $((NUM - 1))`;
do
./main $ARGS &
done

x11vnc --usepw --forever
