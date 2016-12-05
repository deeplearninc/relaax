#!/bin/bash
N=$1
logdir=()
for i in `seq 0 $((N - 1))`;
do
    if ((i)); then
        logdir+=","
    fi
    logdir+="worker_$i:logs/boxing_a3c_${N}_agents/worker_$i"
done    
echo $logdir

source activate server&&tensorboard --logdir $logdir
