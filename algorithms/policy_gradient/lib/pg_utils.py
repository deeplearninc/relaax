import numpy as np


def choose_action(probabilities, exploit):
    if exploit:
        return np.argmax(probabilities)   # need to set greedily param
    return np.random.choice(len(probabilities), p=probabilities)
