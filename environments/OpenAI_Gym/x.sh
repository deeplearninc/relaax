#!/bin/bash

if [ -z "$4" ]
then
    $3 &
    ./main --rlx-server $1 --env $2 &
    x11vnc --usepw --forever
else
    regime=$4
    PIDS=()
    echo $regime

    if [ "$regime" == "display" ]
    then
        ./main --rlx-server $1 --env $2 &
        PIDS+=($!)
        echo $((PIDS[0]))
    elif [ "$regime" -eq "$regime" ]
    then
        for i in `seq 0 $((regime - 1))`;
        do
        ./main --rlx-server $1 --env $2 &
        PIDS+=($!)
        echo $!
        done
    else
        echo "You've passed a wrong arguments... Please, relaunch the docker with the right one"
    fi

    if [ "$regime" == "display" ]
    then
        #x11vnc --usepw --forever && kill -SIGUSR1 $((PIDS[0]))
        (sleep 5; kill -SIGUSR1 $((PIDS[0])))
        x11vnc --usepw --forever
    else
        x11vnc --usepw --forever
    fi
fi
