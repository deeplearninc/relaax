episodes = 10000                    # Number of episodes to pass through train
is_batch_norm = True                # Batch normalization is ON while True (False - batch norm is off)
experiment = 'BipedalWalker-v2'     # Name of the Environment to use in train process

# ddpg
REPLAY_MEMORY_SIZE = 100000          # experience replay memory buffer size [10000]
BATCH_SIZE = 64                     # mini-batch size
GAMMA = 0.99                        # discount factor
is_grad_inverter = True             # grad descent / ascent switcher

# actor_net
LEARNING_RATE = 0.0001
TAU = 0.001                         # discount for updates from local to global
N_HIDDEN_1 = 400                    # amount of hidden neurons at 1st layer
N_HIDDEN_2 = 300                    # amount of hidden neurons at 2nd layer
