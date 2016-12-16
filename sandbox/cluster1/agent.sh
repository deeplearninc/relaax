#!/usr/bin/env bash
python ../../relaax/server/da3c/agent/main.py --config config.yaml --bind localhost:7001 --ps localhost:7000 --log-level WARNING --log-dir boxing_a3c --timeout 120
