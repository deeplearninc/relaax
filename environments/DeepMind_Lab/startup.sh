#!/bin/bash

echo "Initialization...please wait 1 minute"
cd /opt/lab && bazel run :random_agent --define headless=false &>/dev/null
mv /opt/relaax/environments/DeepMind_Lab/random_agent.py /opt/lab/bazel-bin/random_agent.runfiles/org_deepmind_lab/python/random_agent.py

mkdir -p /var/run/sshd

# create an ubuntu user
# PASS=`pwgen -c -n -1 10`
PASS=relaax
# echo "Username: ubuntu Password: $PASS"
id -u ubuntu &>/dev/null || useradd --create-home --shell /bin/bash --user-group --groups adm,sudo ubuntu
echo "ubuntu:$PASS" | chpasswd
sudo -u ubuntu -i bash -c "mkdir -p /home/ubuntu/.config/pcmanfm/LXDE/ \
    && cp /usr/share/doro-lxde-wallpapers/desktop-items-0.conf /home/ubuntu/.config/pcmanfm/LXDE/"

#echo "Run the AGENT......."
#cd /opt/lab/bazel-bin/random_agent.runfiles/org_deepmind_lab
#if [ -z "$2" ]
#  then
#    ./random_agent --rlx-server $1
#else
#    ./random_agent --rlx-server $1 --display $2
#fi

echo $1 > /run_env.rlx-server-url
echo $2 > /run_env.regime
echo $3 > /run_env.level
echo $4 > /run_env.action-size

echo "Initialize Web UI"
cd /web && ./run.py > /var/log/web.log 2>&1 &
nginx -c /etc/nginx/nginx.conf
exec /bin/tini -- /usr/bin/supervisord -n
