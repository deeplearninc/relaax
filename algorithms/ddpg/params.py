episodes = 10**4                    # Number of episodes to pass through train
is_batch_norm = True                # Batch normalization is ON while True (False - batch norm is off)
experiment = 'BipedalWalker-v2'     # Name of the Environment to use in train process

# ddpg
REPLAY_BUFFER_SIZE = 10**6          # experience replay memory buffer size [10000]
REPLAY_START_SIZE = 10**3           # step form which updates begin
BATCH_SIZE = 64                     # mini-batch size
GAMMA = 0.99                        # discount factor

# net
ACTOR_LR = 1e-4                     # Actor learning rate
CRITIC_LR = 1e-3                    # Critic learning rate
TAU = 1e-3                          # discount for updates from local to global
LAYER1_SIZE = 400                   # amount of hidden neurons at 1st layer
LAYER2_SIZE = 300                   # amount of hidden neurons at 2nd layer
L2 = 1e-2                           # Norm
