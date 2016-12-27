#!/usr/bin/env bash
relaax-parameter-server --config config.yaml --bind localhost:7000 --checkpoint-dir checkpoints/boxing_a3c --log-level WARNING
