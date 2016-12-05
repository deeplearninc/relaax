N=$1
AGENT="$2:80"
PIDS=()

source activate client
for i in `seq 0 $((N - 1))`;
do
    echo client $i
    exec python ../../clients/rl_client_ale.py --agent $AGENT --seed $i &>out/client_$i &
    PIDS+=($!)
    sleep 0.5
done

read -p "Press [Enter] key to stop clients..."

for i in `seq $((${#PIDS[@]} - 1)) -1 0`;
do
    echo stop $((PIDS[i]))
    kill -SIGINT $((PIDS[i]))
    # sleep 1
done

sleep 2

ps ax | grep python
