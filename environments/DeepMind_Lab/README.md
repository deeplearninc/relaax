#### How to build your own docker && run environment
```
$ cd path_to_this_dir

# docker build -f Dockerfile -t your_docker_hub_name/image_name ../..
# or you can build without your docker hub username, for example:
$ docker build -f Dockerfile -t lab-vnc-relaax ../..

# docker run -ti -p your_port:6080 docker_image_name
# for example:
$ docker run -ti -p 6080:6080 4skynet/lab
```

You have to wait for 1 minute for some initialization purposes after run.

Then you can navigate in your browser to this address:
```
http://127.0.0.1:6080/vnc.html

# you can type any password firstly
```

Open the terminal > click left bottom icon > Accessories > LXTerminal:
```
$ cd /opt/lab/bazel-bin/random_agent.runfiles/org_deepmind_lab

# ./random_agent --rlx-server rlx_server_ip:port
# for example:
$ ./random_agent --rlx-server 192.168.2.103:7001

# if you want to see the appropriate visual screen
$ ./random_agent --rlx-server 192.168.2.103:7001 --display true
```