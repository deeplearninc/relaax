DIR=boxing_a3c
MASTER=0.0.0.0:7000
exec python master.py --params params.yaml --bind $MASTER --checkpoint-dir checkpoints/$DIR
