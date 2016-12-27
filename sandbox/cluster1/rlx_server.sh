#!/usr/bin/env bash
relaax-rlx-server --config config.yaml --bind localhost:7001 --parameter_server localhost:7000 --log-level WARNING --log-dir logs --timeout 120
