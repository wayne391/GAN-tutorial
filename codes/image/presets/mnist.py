import tensorflow as tf
from utils.ops import *
import numpy as np


class DataConfig:
    height = 28
    width = 28
    channel = 1
    denorm = lambda data: data*255
    path_dataset = 'wayne_mnist_x'

class Generator(object):
    def __init__(self, is_bn=True, name='generator', mtype=0):
        self.is_bn = is_bn
        self.name = name
        self.mtype = mtype

    def __call__(self, in_tensor, reuse=None, is_training=True):

        is_training = None if self.is_bn is False else is_training
        print('\nGenerator type:', self.mtype)
        print('-- G reuse:', reuse, '| bn:', is_training)

        with tf.variable_scope(self.name, reuse=reuse):

            if self.mtype is 0:
                # linear
                l0 = tf.layers.dense(in_tensor, 7*7*32, name='linear')

                # reshape
                l0 = tf.reshape(l0, [-1, 7, 7, 32])  # (7, 7, 32)

                #convnet
                c0 = tf.layers.conv2d_transpose(l0, 128, 4, strides=2, name='c0', padding='same')
                c0 = tf.nn.relu(batch_norm(c0, is_training))
                print('G -', c0.get_shape())         # (14, 14, 256)

                c1 = tf.layers.conv2d_transpose(c0, 64, 4, strides=2, name='c1', padding='same')
                c1 = tf.nn.relu(batch_norm(c1, is_training))
                print('G -', c1.get_shape())         # (28, 28, 128)

                c2 = tf.layers.conv2d_transpose(c1, 1, 4, strides=1, name='c2', padding='same')
                # output = tf.nn.sigmoid(c2)
                # output = c2
                output = tf.nn.leaky_relu(c2)

                print('lrelu')
                print('G -', output.get_shape())     # (28, 28, 1)

            else:
                raise ValueError('unknow type of generator')

            # collect trainable vars
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        return output

class Discriminator(object):
    def __init__(self, is_bn=True, name='critic', mtype=0):
        self.is_bn = is_bn
        self.name = name
        self.mtype = mtype

    def __call__(self, in_tensor, reuse=None, is_training=True, update_collection=None):

        is_training = None if self.is_bn is False else is_training
        print('\nDiscriminator type:', self.mtype)
        print('-- D reuse:', reuse, '| bn:', is_training)

        with tf.variable_scope(self.name, reuse=reuse):

            if self.mtype is 0:
                # convnet

                c1 = tf.layers.conv2d(in_tensor, 64, 4, strides=2, name='c1', padding='same')
                c1 = tf.nn.leaky_relu(batch_norm(c1, is_training))
                print('D -', c1.get_shape()) # (14, 14, 64)

                c2 = tf.layers.conv2d(c1, 128, 4, strides=2, name='c2', padding='same')
                c2 = tf.nn.leaky_relu(batch_norm(c2, is_training))
                print('D -', c2.get_shape()) # (7, 7, 128)

                c3 = tf.layers.conv2d(c2, 256, 3, strides=2, name='c3', padding='valid')
                c3 = tf.nn.leaky_relu(batch_norm(c3, is_training))
                print('D -', c3.get_shape()) # (7, 7, 256)

                # flattern
                c3 = flattern(c3)

                # linear
                output = tf.layers.dense(c3, 1, name='linear')

            elif self.mtype is 1:
                # convnet

                    # block 1
                c1, sig1 = conv2d_sn(in_tensor, 64, 4, strides=2, name='c1',
                     padding='same',update_collection=update_collection)
                c1 = tf.nn.leaky_relu(batch_norm(c1, is_training))
                print('D -', c1.get_shape()) # (14, 14, 64)

                c2, sig2 = conv2d_sn(c1, 128, 4, strides=2, name='c2',
                     padding='same', update_collection=update_collection)
                c2 = tf.nn.leaky_relu(batch_norm(c2, is_training))
                print('D -', c2.get_shape()) # (7, 7, 128)

                c3, sig3 = conv2d_sn(c2, 256, 3, strides=2, name='c3',
                     padding='valid', update_collection=update_collection)
                c3 = tf.nn.leaky_relu(batch_norm(c3, is_training))
                print('D -', c3.get_shape()) # (7, 7, 256)

                # flattern
                c3 = flattern(c3)

                # linear
                output, sig4 = dense_sn(c3, 1, name='linear', update_collection=update_collection)


                # collect trainable vars
                self.side_info = [sig1, sig2, sig3, sig4]

            else:
                raise ValueError('unknow type of discriminator')

            # collect trainable vars
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

            return output