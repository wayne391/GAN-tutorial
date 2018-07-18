import tensorflow as tf
from utils.ops import *
import numpy as np


class DataConfig:
    height = 32
    width = 32
    channel = 3
    num_class = 10
    denorm = lambda data: data*255
    path_x = 'wayne_cifar10_x'
    path_y = 'wayne_cifar10_y'
    y_sample = label_to_onehot(np.concatenate(np.array([[i]*8 for i in range(8)]), 0), 10)

class Generator(object):
    def __init__(self, is_bn=True, name='generator', mtype=0):
        self.is_bn = is_bn
        self.name = name
        self.mtype = mtype

    def __call__(self, in_tensor, y, reuse=None, is_training=True):

        is_training = None if self.is_bn is False else is_training
        print('\nGenerator type:', self.mtype)
        print('-- G reuse:', reuse, '| bn:', is_training)

        with tf.variable_scope(self.name, reuse=reuse):

            if self.mtype is 'cGAN' or self.mtype is 'ACGAN':

                # input concat
                in_tensor = vec_cond_concat(in_tensor, y)
                # linear

                l0 = tf.layers.dense(in_tensor, 4*4*512, name='linear')

                # reshape
                l0 = tf.reshape(l0, [-1, 4, 4, 512])  # (4, 4, 512)
                # l0 = conv_cond_concat(l0, y)
                print('less')

                #convnet
                c0 = tf.layers.conv2d_transpose(l0, 512, 4, strides=2, name='c0', padding='same')
                c0 = tf.nn.relu(batch_norm(c0, is_training))
                print('G -', c0.get_shape())          # (8, 8, 512)

                c1 = tf.layers.conv2d_transpose(c0, 256, 4, strides=2, name='c1', padding='same')
                print('G -', c1.get_shape())          # (16, 16, 256)

                c2 = tf.layers.conv2d_transpose(c1, 128, 4, strides=2, name='c2', padding='same')
                c2 = tf.nn.relu(batch_norm(c2, is_training))
                print('G -', c2.get_shape())          # (32, 32, 128)

                c3 = tf.layers.conv2d_transpose(c2, 3, 3, strides=1, name='c3', padding='same')
                output = tf.nn.sigmoid(c3)
                print('G -', output.get_shape())      # (16, 16, 512)

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

    def __call__(self, in_tensor, y, reuse=None, is_training=True, update_collection=None):

        is_training = None if self.is_bn is False else is_training
        print('\nDiscriminator type:', self.mtype)
        print('-- D reuse:', reuse, '| bn:', is_training)

        with tf.variable_scope(self.name, reuse=reuse):

            # convnet

                # block 1
            c1_1, sig1_1 = conv2d_sn(in_tensor, 64, 3, strides=3, name='c1_1',
                        padding='same', update_collection=update_collection)
            c1_1 = tf.nn.leaky_relu(c1_1)

            c1_2, sig1_2 = conv2d_sn(c1_1, 128, 4, strides=2, name='c1_2',       # (16, 16, 64)
                        padding='same', update_collection=update_collection)
            c1_2 = tf.nn.leaky_relu(c1_2)

            print('less')

                # block 2
            c2_1, sig2_1 = conv2d_sn(c1_2, 128, 3, strides=1, name='c2_1',
                        padding='same', update_collection=update_collection)
            c2_1 = tf.nn.leaky_relu(c2_1)

            c2_2, sig2_2 = conv2d_sn(c2_1, 256, 4, strides=2, name='c2_2',      # (16, 16, 128)
                        padding='same', update_collection=update_collection)
            c2_2 = tf.nn.leaky_relu(c2_2)

                # block 3
            c3_1, sig3_1 = conv2d_sn(c2_2, 256, 3, strides=1, name='c3_1',
                        padding='same', update_collection=update_collection)
            c3_1 = tf.nn.leaky_relu(c3_1)

            c3_2, sig3_2 = conv2d_sn(c3_1, 512, 4, strides=2, name='c3_2',      # (16, 16, 128)
                        padding='same', update_collection=update_collection)
            c3_2 = tf.nn.leaky_relu(c3_2)

                # block 3
            c4, sig4 = conv2d_sn(c3_2, 512, 3, strides=1, name='c4',
                        padding='same', update_collection=update_collection)
            c4 = tf.nn.leaky_relu(c4)
            # c4 = conv_cond_concat(c4, y)

            # flattern
            c4 = flattern(c4)

            # linear
            critic, sig5 = dense_sn(c4, 1, name='l_critic', update_collection=update_collection)



            if self.mtype is 'cGAN':
                # collect trainable vars
                self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

                # collect side info
                self.side_info = [sig1_1, sig1_2, sig2_1, sig2_2, sig3_1, sig3_2, sig4, sig5]

                return critic

            elif self.mtype is 'ACGAN':
                # linear for classification
                l0, sig6 = dense_sn(c4, 128, name='l0', update_collection=update_collection)
                l1 = tf.nn.leaky_relu(batch_norm(l0, is_training))

                classification, sig6 = dense_sn(l1, 10, name='l_class', update_collection=update_collection)

                # collect trainable vars
                self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

                # collect side info
                self.side_info = [sig1_1, sig1_2, sig2_1, sig2_2, sig3_1, sig3_2, sig4, sig5, sig6]

                return critic, classification

            else:
                raise ValueError('unknow type of discriminator')



