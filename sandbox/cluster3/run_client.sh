mkdir -p out
exec python ../../clients/rl_client_ale.py --agent "$1" --game $2 --seed $3 &>out/client_$3
