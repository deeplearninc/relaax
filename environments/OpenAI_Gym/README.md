#### How to build your own docker && run environment
```
$ cd path_to_this_dir

# docker build -f Dockerfile -t your_docker_hub_name/image_name ../..
# or you can build without your docker hub username, for example:
$ docker build -f Dockerfile -t gym-vnc-relaax ../..

# docker run -ti -p rlx_server_ip:port:5900 docker_image_name rlx_server_ip:port environment_name
# you should provide your rlx-server address at least, for example:
$ docker run -ti -p 192.168.2.103:15900:5900 4skynet/gym 192.168.2.103:7001 BipedalWalker-v2
```

#### How to how well agent perform tasks in a client

We have to connect to existing running docker container by its ID
```
# take the ID firstly
$ docker ps
```

You should find the ID under the column CONTAINER ID
```
CONTAINER ID  IMAGE  COMMAND  CREATED  STATUS  PORTS  NAMES
5f7b6dd10cd0  .....  .......  .......  ......  .....  .....
```

Connect to this container by following command:
```
$ docker run -ti docker_image_name bash
```
The last one opens the terminal in interactive regime

You should find the process ID of your running client:
```
$ ps ax | grep python
```

Then you can switch the client regime to display mode by following:
```
$ kill -SIGUSR1 process_num
```

Then you can connect to client's visual via your VNC client with:
```
For example:

Server: 192.168.2.103:15900
Passwd: relaax
Color depth: True color (24 bit)
```

You have to use this password and depth to see the visual output

You can switch the client regime to display off by following again:
```
$ kill -SIGUSR1 process_num
```