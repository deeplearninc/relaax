#!/bin/bash

path_to_rom="/roms/${2}.bin"
echo $path_to_rom

./main --rlx-server $1 --rom "$path_to_rom"
