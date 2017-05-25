#!/bin/bash

# Environment variables, which you can set to docker with -e
: ${VNC_PASS:=relaax}
: ${X11_DISPLAY_NO:=99}

# VNC server environment: set password
mkdir ~/.vnc
x11vnc -storepasswd "$VNC_PASS" ~/.vnc/passwd

xvfb-run -n "$X11_DISPLAY_NO" -s "-screen 0 1280x720x24" ./x.sh "$@"
