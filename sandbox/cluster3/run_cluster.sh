#!/usr/bin/env bash
N=$1
DIR=boxing_a3c
MASTER=localhost:7000
AGENT=localhost:7001
PIDS=()

echo master
source activate server&&exec python master.py --params params.yaml --bind $MASTER --checkpoint-aws-s3 dl-checkpoints $DIR --aws-keys aws-keys.yaml --log-level WARNING &>out/master &
PIDS+=($!)
sleep 1

echo agent
source activate server&&exec python agent.py --params params.yaml --bind $AGENT --master $MASTER --log-dir logs/$DIR --log-level WARNING &>out/agent &
PIDS+=($!)
sleep 1

for i in `seq 0 $((N - 1))`;
do
    echo client $i
    source activate client&&exec python ../../clients/rl_client_ale.py --agent $AGENT --game boxing --seed $i &>out/client_$i &
    PIDS+=($!)
    sleep 1
done

read -p "Press [Enter] key to stop cluster..."

for i in `seq $((${#PIDS[@]} - 1)) -1 0`;
do
    echo stop $((PIDS[i]))
    kill -SIGINT $((PIDS[i]))
    sleep 1
done

sleep 2

ps ax | grep python
