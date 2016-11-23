AGENT=35.164.194.133:80

source activate client&&exec python ../../clients/rl_client_ale.py --agent $AGENT --seed 0
