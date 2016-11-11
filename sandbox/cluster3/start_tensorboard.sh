#!/bin/bash
n=$1
logdir=()
for i in `seq 0 $((n - 1))`;
do
    if ((i)); then
        logdir+=","
    fi
    logdir+="worker_$i:logs/boxing_a3c_1threads/worker_$i"
done    
echo $logdir

source activate server&&tensorboard --logdir $logdir
