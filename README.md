# REinforcement Learning Algorithms, Autoscaling and eXchange (RELAAX)

We expose state-of-the-art reinforcement learning algorithms in easy to use RELAAX framework. RELAAX allows your to scale training dinamically by running cluster of RL Agents on any of the popular clouds and connecting RL Environments over GRPC based [Clients-Agents eXchange protocol](#clients-agents-exchange-protocol).

* [RELAAX Client](#relaax-clients) is wrapping details of the [Clients-Agents eXchange protocol](#clients-agents-exchange-protocol) implementation and exposes simple API to be used to exchange Stata, Revard, and Actions between scalable RL Server and Environment. 

* [RELAAX Server](#relaax-server) allow developers to run RL Agents locally or at scale on popular cloud platforms. See more details below.

* RELAAX provides implementations of the popular [RL algorithms](#algorithms) to simplify RL application(s) development and research. 

* RELAAX comes with set of Terraform scripts which allow you to build your cluster on AWS, GCP, and Azure

## Contents
- [System Architecture](#system-architecture)
- [RELAAX Clients](#relaax-clients)
 - [Clients-Agents eXchange protocol](#clients-agents-exchange-protocol)
 - [Supported Environments](#supported-environments)
- [RELAAX Server](#relaax-server)
 - [Parameter Server](#parameter-server)
 - [Workers](#workers)
 - [Visualization](#visualization)
- [Algorithms](#algorithms)
 - [Destributed A3C](#destributed-a3c)
 - [Other Algorithms](#other-algorithms)
- [Repository Overview](#repository-overview)

## [System Architecture](#contents)

## [RELAAX Clients](#contents)
Client is small library which could be used with the Environment implemented in many popular laguages or embedded into specialised hardware systems. Currently client support ALE, OpenAI gym, and OpenAI Universe Environments. Later on we are planning to implement client code in C/C++, Ruby, GO, etc. to simplify integration of other inveronments.

###  [Clients-Agents eXchange protocol](#contents)

Clients-Agents eXchange protocol is a simple protocol implemented over GRPC. It allow to send State of the Environment and Revard to the Server and deliver Action from the Agent to the Environment. 

![img](resources/protocol-flow.png)

### [Supported Environments](#contents)

* [ALE](/clients/rl-client-ale)
* [OpenAI Gym](/clients/rl-client-gym)
 * [Classic Control](https://gym.openai.com/envs#classic_control)
 * [Atari Games](https://gym.openai.com/envs#atari)
 * [Walkers, Landers & Racing](https://gym.openai.com/envs##box2d)
* [OpenAI Universe](https://universe.openai.com/)

## [RELAAX Server](#contents)
### [Parameter Server](#contents)
### [Workers](#contents)
### [Visualization](#contents)

## [Algorithms](#contents)
 
### [Destributed A3C](#contents)
Inspired by original [paper](https://arxiv.org/abs/1602.01783) - Asynchronous Methods for Deep Reinforcement Learning from [DeepMind](https://deepmind.com/)

##### Destributed A3C Architecure
![img](resources/DA3C.png)

- Parameter (Master) Server holds the global parameters of training process
such as: global neural network weights, state of optimizer and gradients slots
- Agents (Learners) perform the main training loop: make a copy of global
network, collect batch from client's data and compute gradients wrt loss,
which transmits to master server to apply
- Clients (Environments) connect to Agent (more than one client can connect)
and send their data to it: state, reward and terminal (if exist)

##### Performance on some of the Atari Environments
Breakout with DA3C-FF and 8 parallel agents: score performance is similar to DeepMind [paper](https://arxiv.org/pdf/1602.01783v2.pdf#19)
![img](resources/Breakout-8th-80mil.png "Breakout")

Breakout with DA3C-FF and 8 parallel agents: ih this case we outperforms significantly DeepMind, but
we have some instability in training process (anyway DeepMind shows only 34 points after 80mil steps)
![img](resources/Boxing-8th-35mil.png "Boxing")

##### Compute Performance with different amount of clients and node types (AWS):

| Node Type  | Number of clients | Performance       |
| ---------- |:-----------------:| -----------------:|
| m4.xlarge  |          32       | 99 steps per sec  |
| m4.xlarge  |          64       | 171 steps per sec |    
| m4.xlarge  |          48       | 167 steps per sec |
| c4.xlarge  |          48       | 169 steps per sec |
| c4.xlarge  |          64       | 207 steps per sec |
| c4.xlarge-m4.xlarge | 64       | 170 steps per sec |
| c4.xlarge-m4.xlarge | 96       | 167 steps per sec |
| c4.xlarge-m4.xlarge | 128      | 177 steps per sec |
| c4.2xlarge |          232      | 232 steps per sec |
| c4.2xlarge |          271      | 271 steps per sec |
<br><br>

    
### [Other Algorithms](#contents)
These other algorithms we are working on and planning to make distributed versions of: 

* TRPO-GAE
Inpired by:
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

* PPO with L-BFGS (similar to TRPO)
Inpired by:
* CEM
Inpired by:
* DDPG
Inpired by:
* Distributed DQN
Inpired by:
 - [Gorila](http://) 
 
## [Repository Overview](#contents)
  - [Algorithms]()
    - [Distributed A3C]()
      - Parameter Server
        ... components of the A3C PS
        - global_policy.py 
        etc.
      - Agent
        ... components of the A3C agent
        - actor_critic_network.py
        etc...
  - [Server]()
    - Parameter Server
      - grpc_interface.py
      - global_policy_runner.py
      - metrics_server.py
    - Worker
      - grpc_server.py
      - agent_runner.py
      - metrics_api.py
  - [Clients]()
    - python
      - Common 
        - grpc_interface.py
      - [ALE]()
        - client.py
        - Dockerfile
      - [OpenAI gym]()
        - client.py
        - Dockerfile
      - OpenAI Universe
        - client.py
        - Dockerfile
