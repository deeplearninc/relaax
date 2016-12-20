#!/usr/bin/env bash
python ../../relaax/server/rlx_server/main.py --config config.yaml --bind localhost:7001 --parameter_server localhost:7000 --log-level WARNING --log-dir logs --timeout 120
