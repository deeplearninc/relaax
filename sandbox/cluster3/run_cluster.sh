source activate server&&exec python ps.py &>out/ps &
PS_PID=$!

sleep 1

source activate server&&exec /home/sv/anaconda2/envs/server/bin/gunicorn -k flask_sockets.worker -b localhost:8000 "worker:main(0)" &>out/worker_0 &
WORKER_0_PID=$!

sleep 1

source activate server&&exec /home/sv/anaconda2/envs/server/bin/gunicorn -k flask_sockets.worker -b localhost:8001 "worker:main(1)" &>out/worker_1 &
WORKER_1_PID=$!

sleep 1

source activate client&&exec python ../../clients/rl_client_ale.py --host localhost --port 8000 --agents 8 &>out/client_0 &
CLIENT_0_PID=$!

sleep 1

source activate client&&exec python ../../clients/rl_client_ale.py --host localhost --port 8001 --agents 8 &>out/client_1 &
CLIENT_1_PID=$!


read -p "Press [Enter] key to stop cluster..."

kill -SIGINT $CLIENT_1_PID

sleep 1
kill -SIGINT $CLIENT_0_PID

sleep 1
kill -SIGINT $WORKER_1_PID

sleep 1
kill -SIGINT $WORKER_0_PID

sleep 1
kill -SIGINT $PS_PID

sleep 3

ps ax | grep python
