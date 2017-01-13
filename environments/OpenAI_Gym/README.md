```
cd path_to_this_dir
$ docker build -f Dockerfile -t 4skynet/gym ../..
$ docker run -ti 4skynet/gym 192.168.2.103:7001 BipedalWalker-v2
```