import numpy as np
import tensorflow as tf
import tensorlayer as tl


def conv2d(x, num_filter, kernel_size=(3, 3), stride_size=(1, 1), bias_val=0.01, stddev=0.01, padding='VALID', activation='relu', name='conv2d'):
    k_w, k_h = kernel_size
    s_w, s_h = stride_size
    with tf.variable_scope(name) as score:
        w = tf.get_variable(name='w', shape=[k_w, k_h, x.get_shape()[-1], num_filter],
                           initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable(name='b', shape=[num_filter], initializer=tf.constant_initializer(bias_val))
        if activation == 'sigmoid':
            tf.nn.sigmoid(tf.nn.conv2d(x, w, strides=[1, s_w, s_h, 1], padding=padding) + b)
        elif activation == 'tanh':
            tf.nn.tanh(tf.nn.conv2d(x, w, strides=[1, s_w, s_h, 1], padding=padding) + b)
        else: # relu
            return tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, s_w, s_h, 1], padding=padding) + b)
    
def max_pool_2d(x, kernel_size=(3, 3), name='max_pool_2d'):
    k_w, k_h = kernel_size
    return tf.nn.max_pool(x, ksize=[1, k_w, k_h, 1], strides=[1, k_w, k_h, 1], padding='SAME', name=name)

def dense(x, output_dim, bias_val=0.001, stddev=0.1, activation=None, name='dense'):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[x.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.constant_initializer(bias_val))
        if activation == 'relu':
            return tf.nn.relu(tf.add(tf.matmul(x, w), b))
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(tf.add(tf.matmul(x, w), b))
        elif activation == 'tanh':
            return tf.nn.tanh(tf.add(tf.matmul(x, w), b))
        else:
            return tf.add(tf.matmul(x, w), b)

def flatten(x, name='flatten'):
    with tf.name_scope(name) as score:
        return tf.reshape(x, shape=[-1, tf.reduce_prod(x.get_shape().as_list()[1:])])
    
def dropout(x, keep_prob=1.0, name='dropout'):
    return tf.nn.dropout(x, keep_prob=keep_prob, name=name)