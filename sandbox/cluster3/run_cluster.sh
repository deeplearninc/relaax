N=16
PIDS=()

echo ps
source activate server&&exec python ps.py &>out/ps &
PIDS+=($!)
sleep 1

for i in `seq 0 $((N - 1))`;
do
    echo worker $i
    source activate server&&exec gunicorn -k flask_sockets.worker -b localhost:$((8000 + i)) "worker:main($i)" &>out/worker_$i &
    PIDS+=($!)
    sleep 1
done

for i in `seq 0 $((N - 1))`;
do
    echo client $i
    source activate client&&exec python ../../clients/rl_client_ale.py --host localhost --port $((8000 + i)) --agents 1 &>out/client_$i &
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
