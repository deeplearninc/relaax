#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

(cd $DIR&&python -m grpc.tools.protoc -I. --python_out=. --grpc_python_out=. bridge.proto)
