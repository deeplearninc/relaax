DIR=boxing_a3c
MASTER=10.0.1.184:7000
AGENT=0.0.0.0:80

exec python agent.py --params params.yaml --bind $AGENT --master $MASTER --log-dir logs/$DIR --log-level INFO 
