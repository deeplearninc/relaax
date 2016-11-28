How to load cluster.


Run cluster:

1. clone git@github.com:deeplearninc/infrastrucure.git

2. checkout v1 branch

3. navigate to rl-cluster subdirectory

3. run `terraform apply`

4. wait for completion and copy cluster_public_address


Run clients:

1. clone git@github.com:deeplearninc/rl-server.git

2. checkout listen_accept_fork

3. navigate to sandbox/cluster3 subdirectory

4. run `python run_clients.py --agent <cluster_public_address>:80` 

5. inside the app run "+" commmand and wait for "clients: 1" message

6. add clients running "10+" commands for example, remove clients running "10-" commands, watch client console running "t" command.
