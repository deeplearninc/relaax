p:
	bash -c 'source activate server&&python ps.py'

w0:
	bash -c 'source activate server&&/home/sv/anaconda2/envs/server/bin/gunicorn -k flask_sockets.worker -b 127.0.0.1:8000 "worker:main(0)"'

w1:
	bash -c 'source activate server&&/home/sv/anaconda2/envs/server/bin/gunicorn -k flask_sockets.worker -b 127.0.0.1:8001 "worker:main(1)"'

c0:
	bash -c 'source activate client&&python ../../clients/rl_client_ale.py --host localhost --port 8000 --lstm no'

c1:
	bash -c 'source activate client&&python ../../clients/rl_client_ale.py --host localhost --port 8001 --lstm no'
