import numpy as np 
import tensorflow as tf

import logging
log = logging.getLogger("policy_gradient")

from relaax.common.algorithms.python.decorators import define_scope, define_input

from pg_config import config, options 

### Holder for variables representing 
### weights of the policy gradient NN  
class Weights(object):
    def __init__(self,input_size,output_size,hidden_layers):
        # Initialize network weights composition
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        ### Build TF graph
        self.weights

    @define_scope(initializer=tf.contrib.layers.xavier_initializer())
    def weights(self):
        ### weights of the NN
        weights = []

        def add_layer(name,input_size,layer_size):
            return weights.append(tf.get_variable(name, shape=[input_size,layer_size]))

        # input layer weights
        add_layer('W0',self.input_size,self.hidden_layers[0])

        # hidden layer weights
        nlayers = len(self.hidden_layers)
        for i in range(0, nlayers-1):
            add_layer('W%d'%(i+1),self.hidden_layers[i],self.hidden_layers[i+1])

        ### output layer weights
        add_layer('W%d'%nlayers,self.hidden_layers[-1],self.output_size)    

        return weights

### Weights of the policy NN are shared across 
### all agents and stored on  parameter server
class SharedParameters(Weights):

    def __init__(self):
        ### Build TF graph
        super(SharedParameters,self).__init__(
            config.state_size,config.action_size,config.hidden_layers)
        self.gradients
        self.apply_gradients

    @define_scope
    def gradients(self):
        ### placeholders to apply gradients to shared parameters 
        return [tf.placeholder(v.dtype, v.get_shape()) for v in self.weights]

    @define_scope
    def apply_gradients(self):
        ## apply gradients to weights
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        return optimizer.apply_gradients(zip(self.gradients, self.weights))

### Policy NN run by Agent(s)
class PolicyNN(Weights):
    
    def __init__(self):
        ### Build TF graph
        super(PolicyNN,self).__init__(
            config.state_size,config.action_size,config.hidden_layers)
        self.state
        self.action
        self.discounted_reward
        self.shared_wights
        self.policy
        self.partial_gradients
        self.assign_weights

    @define_input
    def state(self):
        return tf.placeholder(tf.float32, [None, config.state_size])

    @define_input
    def action(self):
        return tf.placeholder(tf.float32, [None, config.action_size])

    @define_input
    def discounted_reward(self):
        return tf.placeholder(tf.float32)

    @define_input
    def shared_wights(self):
        ### placeholders to apply weights to shared parameters 
        return [tf.placeholder(v.dtype, v.get_shape()) for v in self.weights]

    @define_scope
    def policy(self):
        layer = tf.nn.relu(tf.matmul(self.state, self.weights[0]))
        for i in range(1, len(self.weights)-1):
            layer = tf.nn.relu(tf.matmul(layer, self.weights[i]))
        return tf.nn.softmax(tf.matmul(layer, self.weights[-1]))

    @define_scope
    def loss(self):
        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.
        log_like = tf.log(tf.reduce_sum(tf.multiply(self.action, self.policy)))
        return -tf.reduce_mean(log_like*self.discounted_reward)

    @define_scope
    def partial_gradients(self):
        return tf.gradients(self.loss, self.weights)

    @define_scope
    def assign_weights(self):
        return tf.group(*[tf.assign(v, p) for v, p in zip(self.weights, self.shared_wights)])

if __name__ == '__main__':
    def build_and_show_graph():
        with tf.variable_scope('ps'):
            SharedParameters()
        with tf.variable_scope('agent'):
            PolicyNN()
        log_dir = options.get("agent/log_dir","log/")
        log.info("Writing TF summary to %s. Please use tensorboad to watch." % log_dir)
        tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    build_and_show_graph()
