p:
	bash -c 'source activate server&&exec python ps.py'

w0:
	bash -c 'source activate server&&exec /home/sv/anaconda2/envs/server/bin/gunicorn -k flask_sockets.worker -b localhost:8000 "worker:main(0)"'

w1:
	bash -c 'source activate server&&exec /home/sv/anaconda2/envs/server/bin/gunicorn -k flask_sockets.worker -b localhost:8001 "worker:main(1)"'

c0:
	bash -c 'source activate client&&exec python ../../clients/rl_client_ale.py --host localhost --port 8000 --agents 4'

c1:
	bash -c 'source activate client&&exec python ../../clients/rl_client_ale.py --host localhost --port 8001 --agents 4'
