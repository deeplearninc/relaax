#!/bin/bash

# DEFAULT VALUES
NUM=1    # number of clients to run
ARGS=""  # concat all args to one string

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
    ARGS+="--env $ENV "
    shift # rm value
    ;;
    -e=*|--env=*)
    ENV="${key#*=}"
    ARGS+="--env $ENV "
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
    -l|--limit)
    shift # rm argument
    LIMIT="$1"
    ARGS+="--limit $LIMIT "
    shift # rm value
    ;;
    -l=*|--limit=*)
    LIMIT="${key#*=}"
    ARGS+="--limit $LIMIT "
    shift # rm argument=value
    ;;
    -r|--rnd)
    shift # rm argument
    RND="$1"
    ARGS+="--rnd $RND "
    shift # rm value
    ;;
    -r=*|--rnd=*)
    RND="${key#*=}"
    ARGS+="--rnd $RND "
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
    shift # rm argument
    ;;
    *) # unknown option
    shift
    ;;
esac
done

echo "ARGS = ${ARGS}"

PIDS=()
for i in `seq 0 $((NUM - 1))`;
do
./main $ARGS &
PIDS+=($!)
done

if [ "$VISUAL_OUTPUT" == "YES" ]
then
(sleep 5; kill -SIGUSR1 $((PIDS[0])))
fi

x11vnc --usepw --forever
