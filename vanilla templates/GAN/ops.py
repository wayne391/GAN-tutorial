from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def dense(x, output_dim, bias_val = 0.0, stddev=0.1, name='dense'):
    with tf.variable_scope(name):
        W = tf.get_variable('W', 
            [ x.get_shape()[-1],output_dim], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias_val))
        return tf.matmul(x, W) + b

def deconv2d(x, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, bias_val = 0.0, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('W', 
            [ k_h, k_w, output_shape[-1], x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', output_shape[-1], initializer=tf.constant_initializer(bias_val))
        
        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, d_h, d_w, 1], padding = 'SAME')
        return tf.add(conv, b)

def conv2d(x,output_dim, k_h=5, k_w=5,  d_h=2, d_w=2, bias_val = 0.0, stddev=0.02, name='conv2d'):
     with tf.variable_scope(name):
        W = tf.get_variable('W', 
            [k_h,  k_w, x.get_shape()[-1] ,output_dim], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias_val))
        conv = tf.nn.conv2d(x, W, strides=[1, d_h, d_w, 1], padding = 'SAME')
        return tf.add(conv, b)

def lrelu(x, leak=0.1, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)