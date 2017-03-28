import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense as FullyConnected
from keras.optimizers import Adam

from relaax.common.algorithms.decorators import define_scope, define_input
