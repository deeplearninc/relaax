#!/usr/bin/env bash
python ../../relaax/server/ps/main.py --config config.yaml --bind localhost:7000 --checkpoint-aws-s3 dl-checkpoints boxing_a3c --aws-keys aws-keys.yaml --log-level WARNING
