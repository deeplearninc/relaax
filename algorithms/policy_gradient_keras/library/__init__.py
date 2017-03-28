from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import moves

import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense as FullyConnected
import keras.backend

from relaax.common.algorithms.decorators import define_scope, define_input
