import tensorflow as tf
from utils.ops import *
import numpy as np


class DataConfig:
    height = 96
    width = 84
    channel = 2
    path_dataset = 'wayne_tab_x_bar'
    denorm = lambda data: data*255
    colormap = np.array([[.7, .5, 0.],
                         [0., .5, .7]])
    thres = 0.5

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
                l0 = tf.layers.dense(in_tensor, 8*512, name='linear')

                # reshape
                l0 = tf.reshape(l0, [-1, 8, 1, 512]) # (8, 1, 512)

                #convnet
                c0 = tf.layers.conv2d_transpose(l0, 256, [4, 1], strides=[2, 1], name='c0', padding='same')
                c0 = tf.nn.relu(batch_norm(c0, is_training))
                print('G -', c0.get_shape())         # (16, 1, 256)

                c1 = tf.layers.conv2d_transpose(c0, 256, [1, 12], strides=[1, 12], name='c1', padding='same')
                c1 = tf.nn.relu(batch_norm(c1, is_training))
                print('G -', c1.get_shape())         # (16, 12, 256)

                c2 = tf.layers.conv2d_transpose(c1, 128, [4, 1], strides=[2, 1], name='c2', padding='same')
                c2 = tf.nn.relu(batch_norm(c2, is_training))
                print('G -', c2.get_shape())         # (32, 12, 128)

                c3 = tf.layers.conv2d_transpose(c2, 128, [1, 7], strides=[1, 7], name='c3', padding='same')
                c3 = tf.nn.relu(batch_norm(c3, is_training))
                print('G -', c3.get_shape())         # (32, 84, 128)

                c4 = tf.layers.conv2d_transpose(c3, 64, [3, 1], strides=[3, 1], name='c4', padding='same')
                c4 = tf.nn.relu(batch_norm(c4, is_training))
                print('G -', c4.get_shape())         # (96, 84, 64)

                c5 = tf.layers.conv2d_transpose(c4, 2, [1, 1], strides=[1, 1], name='c5', padding='same')
                output = tf.nn.relu(batch_norm(c5, is_training))
                print('G -', output.get_shape())     # (96, 84, 2)

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
                # convnetc2

                c1 = tf.layers.conv2d(in_tensor, 64, [3, 1], strides=[3, 1], name='c1', padding='same')
                c1 = tf.nn.leaky_relu(batch_norm(c1, is_training))
                print('D -', c1.get_shape())  # (32, 84, 64)

                c2 = tf.layers.conv2d(c1, 128, [1, 7], strides=[1, 7], name='c2', padding='same')
                c2 = tf.nn.leaky_relu(batch_norm(c2, is_training))
                print('D -', c2.get_shape())  # (32, 12, 128)

                c3 = tf.layers.conv2d(c2, 128, [4, 1], strides=[2, 1], name='c3', padding='same')
                c3 = tf.nn.leaky_relu(batch_norm(c3, is_training))
                print('D -', c3.get_shape())  # (16, 12, 128)

                c4 = tf.layers.conv2d(c3, 256, [4, 1], strides=[2, 1], name='c4', padding='same')
                c4 = tf.nn.leaky_relu(batch_norm(c4, is_training))
                print('D -', c4.get_shape())  # (8, 12, 128)

                c5 = tf.layers.conv2d(c4, 256, [1, 12], strides=[1, 12], name='c5', padding='same')
                c5 = tf.nn.leaky_relu(batch_norm(c5, is_training))
                print('D -', c5.get_shape())  # (8, 1, 128)

                c6 = tf.layers.conv2d(c5, 512, [4, 1], strides=[1, 1], name='c6', padding='same')
                c6 = tf.nn.leaky_relu(batch_norm(c6, is_training))
                print('D -', c6.get_shape())  # (8, 1, 512)

                # flattern
                c6 = flattern(c6)

                # linear
                output = tf.layers.dense(c6, 1, name='l1')

            elif self.mtype is 1:
                # convnetc2

                c1, sig1 = conv2d_sn(in_tensor, 64, [3, 1], strides=[3, 1], name='c1',
                            padding='SAME', update_collection=update_collection)
                c1 = tf.nn.leaky_relu(batch_norm(c1, is_training))
                print('D -', c1.get_shape())  # (32, 84, 64)

                c2, sig2 = conv2d_sn(c1, 128, [1, 7], strides=[1, 7], name='c2',
                            padding='SAME', update_collection=update_collection)
                c2 = tf.nn.leaky_relu(batch_norm(c2, is_training))
                print('D -', c2.get_shape())  # (32, 12, 128)

                c3, sig3 = conv2d_sn(c2, 128, [4, 1], strides=[2, 1], name='c3',
                            padding='SAME', update_collection=update_collection)
                c3 = tf.nn.leaky_relu(batch_norm(c3, is_training))
                print('D -', c3.get_shape())  # (16, 12, 128)

                c4, sig4 = conv2d_sn(c3, 256, [4, 1], strides=[2, 1], name='c4',
                            padding='SAME', update_collection=update_collection)
                c4 = tf.nn.leaky_relu(batch_norm(c4, is_training))
                print('D -', c4.get_shape())  # (8, 12, 128)

                c5, sig5 = conv2d_sn(c4, 256, [1, 12], strides=[1, 12], name='c5',
                            padding='SAME', update_collection=update_collection)
                c5 = tf.nn.leaky_relu(batch_norm(c5, is_training))
                print('D -', c5.get_shape())  # (8, 1, 128)

                c6, sig6 = conv2d_sn(c5, 512, [4, 1], strides=[1, 1], name='c6',
                            padding='SAME', update_collection=update_collection)
                c6 = tf.nn.leaky_relu(batch_norm(c6, is_training))
                print('D -', c6.get_shape())  # (8, 1, 512)

                # flattern
                c6 = flattern(c6)

                # linear
                output, sig7 = dense_sn(c6, 1, name='l1')

            else:
                raise ValueError('unknow type of discriminator')

            # collect trainable vars
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

            return output