#!/usr/bin/env bash
python ../../relaax/server/agent/main.py --config config.yaml --bind localhost:7001 --ps localhost:7000 --log-level WARNING --log-dir logs --timeout 120
