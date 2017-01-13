#!/bin/bash

# env vars
: ${VNC_PASS:=relaax}
: ${X11_DISPLAY_NO:=99}
: ${X11_WINDOW_MANAGER:=ratpoison}

# VNC server environment: set password
mkdir ~/.vnc
x11vnc -storepasswd "$VNC_PASS" ~/.vnc/passwd

xvfb-run -n "$X11_DISPLAY_NO" -s "-screen 0 1280x720x24" ./x.sh $1 $2 "$X11_WINDOW_MANAGER"