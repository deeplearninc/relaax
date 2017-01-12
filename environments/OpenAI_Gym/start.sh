#!/bin/bash

xvfb-run -s "-screen 0 1280x720x24" python main --rlx-server $1 --env $2