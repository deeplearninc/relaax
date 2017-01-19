#!/bin/bash

path_to_rom="/roms/${2}.bin"
echo "$path_to_rom"
regime=$3

if [ -z "$regime" ]
then
    ./main --rlx-server $1 --rom "$path_to_rom"
elif [ "$3" == "display" ]
then
    # Environment variables, which you can set to docker with -e
    : ${VNC_PASS:=relaax}
    : ${X11_DISPLAY_NO:=99}
    : ${X11_OUT:=1024x768x24}
    : ${X11_WINDOW_MANAGER:=ratpoison}

    # VNC server environment: set password
    mkdir ~/.vnc
    x11vnc -storepasswd "$VNC_PASS" ~/.vnc/passwd

    xvfb-run -n "$X11_DISPLAY_NO" -s "-screen 0 $X11_OUT" ./x.sh $1 "$path_to_rom" "$X11_WINDOW_MANAGER"
elif [ "$3" -eq "$3" ]
then
    for i in `seq 0 $((regime - 1))`;
    do
    ./main --rlx-server $1 --rom "$path_to_rom" &
    echo $!
    done
    read -p "Press [Enter] key to stop the docker..."
else
    echo "You've passed a wrong arguments... Please, relaunch the docker with the right one"
fi
