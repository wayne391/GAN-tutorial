import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from ops import *
from utils import *


class MLPGAN(object):
    def __init__(self, sess, image_size=784, hidden_layer_size=[150, 300], keep_prob=1.0,
                 d_lr=0.0003, g_lr=0.0001, epoch=500, batch_size=256, z_dim=100,
                 data_dir=None, checkpoint_dir=None, output_dir=None, log_dir=None,
                 model_name=None, sample_duration=5, is_restore=False):
        self._sess = sess
        self._image_size = image_size
        self._hidden_layer_size = hidden_layer_size
        self._keep_prob = keep_prob
        self._d_lr = d_lr
        self._g_lr = g_lr
        self._epoch = epoch
        self._batch_size = batch_size
        self._z_dim = z_dim
        self._data_dir = data_dir
        self._checkpoint_dir = checkpoint_dir
        self._output_dir = output_dir
        self._log_dir = log_dir
        self._model_name = model_name
        self._sample_duration = sample_duration
        self._is_restore = is_restore
        self._x_train, self._y_train, self._x_test, self._y_test = read_mnist_data(self._data_dir)
        self._x_train = np.concatenate((self._x_train, self._x_test), axis=0)
        self._x_train = 2 * self._x_train.astype(np.float32) - 1
        self._img_cache_fixed = []
        self._img_cache_random = []
        self.build_model()
    
    
    def generator(self, z, reuse=False, name='Generator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            h1 = dense(z,  self._hidden_layer_size[0], activation='relu', name='h_1')
            h2 = dense(h1, self._hidden_layer_size[1], activation='relu', name='h_2')
            h3 = dense(h2, self._image_size, activation='tanh', name='h_3')
            return h3
    
    
    def discriminator(self, x, reuse=False, name='Discriminator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            h1 = dense(x,  self._hidden_layer_size[1], activation='relu', name='h_1')
            h1 = dropout(h1, self._keep_prob, name='dp_1')
            h2 = dense(h1, self._hidden_layer_size[0], activation='relu', name='h_2')
            h2 = dropout(h2, self._keep_prob, name='dp_2')
            h3 = dense(h2, 1, activation='sigmoid', name='h_3')
            return h3
        
        
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        self.z = tf.placeholder(tf.float32, shape=[None, self._z_dim], name='z')
        
        self.Gz = self.generator(self.z, reuse=False)
        self.Dx = self.discriminator(self.x, reuse=False)
        self.Dg = self.discriminator(self.Gz, reuse=True)
        
        # several questions here:
        #   1. why the loss functions aren't the same with those on the original GAN paper?
        #   2. why the loss functions aren't the same with those on medium post? (URL: https://goo.gl/GQoTyo)
        with tf.name_scope('loss_func'):
            self.d_loss = -(tf.log(self.Dx) + tf.log(1. - self.Dg))
            self.g_loss = (-tf.log(self.Dg))
        
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'Discriminator' in var.name]
        g_vars = [var for var in tvars if 'Generator' in var.name]
        
        self.d_optimizer = tf.train.AdamOptimizer(self._d_lr).minimize(self.d_loss, var_list=d_vars)
        self.g_optimizer = tf.train.AdamOptimizer(self._g_lr).minimize(self.g_loss, var_list=g_vars)
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        
        self.z_his = tf.summary.histogram('z', self.z)
        self.dx_his = tf.summary.histogram('Dx', self.Dx)
        self.dg_his = tf.summary.histogram('Dg', self.Dg)
        self.d_loss_his = tf.summary.histogram('d_loss', self.d_loss)
        self.g_loss_his = tf.summary.histogram('g_loss', self.g_loss)
        self._saver = tf.train.Saver()
        
        
    def train(self):
        init = tf.global_variables_initializer()
        self._sess.run(init)
        
        if self._is_restore:
            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            self._saver.restore(self._sess, chkpt_fname)
    
        batches = np.ceil(self._x_train.shape[0] / self._batch_size).astype(int)
        z_val = np.random.normal(0.0, 1.0, size=(self._batch_size, self._z_dim))
        
        self.d_sum_his = tf.summary.merge([self.z_his, self.dx_his, self.d_loss_his])
        self.g_sum_his = tf.summary.merge([self.z_his, self.dg_his, self.g_loss_his])
        self._writer = tf.summary.FileWriter(self._log_dir, graph=self._sess.graph)
        
        self.progress_bar = ProgressBar(0, batches, length=30, is_percent=False)
        for epoch in range(self._sess.run(self.global_step), self._epoch): # plus
            start_time = time.time()
            for i in range(batches):
                x_batch = self._x_train[i * self._batch_size : (i+1) * self._batch_size]
                z_batch_size = x_batch.shape[0]
                z_batch = np.random.normal(0.0, 1.0, size=(z_batch_size, self._z_dim))
    
                _, sum_d_his = self._sess.run([self.d_optimizer, self.d_sum_his], feed_dict={self.x: x_batch, self.z: z_batch})
                _, sum_g_his = self._sess.run([self.g_optimizer, self.g_sum_his], feed_dict={self.x: x_batch, self.z: z_batch})
                prefix = ("Epoch: %5d/%d" % (epoch + 1, self._epoch))
                suffix = ("Elapsed Time: %.3f" % (time.time() - start_time))
                self.progress_bar.update(i+1, prefix=prefix, suffix=suffix)
            
            if epoch % self._sample_duration == 0 or epoch == self._epoch - 1:
                gx_val = self._sess.run(self.Gz, feed_dict={self.z: z_val})
                fixed_img = show_result(gx_val, os.path.join(self._output_dir, "fixed-sample-%s.jpg" % (epoch)))
                self._img_cache_fixed += [fixed_img]
                
                z_random_val = np.random.normal(0.0, 1.0, size=(self._batch_size, self._z_dim))
                gx_random_val = self._sess.run(self.Gz, feed_dict={self.z: z_random_val})
                random_img = show_result(gx_random_val, os.path.join(self._output_dir, "random-sample-%s.jpg" % (epoch)))
                self._img_cache_random += [random_img]
                self.save()
                
            self._sess.run(tf.assign(self.global_step, epoch + 1))
            self._writer.add_summary(sum_d_his, epoch)
            self._writer.add_summary(sum_g_his, epoch)
                
            
    def test(self):
        chkpt_fname = tf.train.latest_checkpoint(self._output_dir)
        
        init = tf.global_variables_initializer()
        self._sess.run(init)
        self._saver.restore(self._sess, chkpt_fname)
        
        z_test_val = np.random.uniform(low=0.0, high=1.0, size=(self._batch_size, self._z_dim))
        gx_test_val = self._sess.run(self.Gz, feed_dict={self.z: z_test_val})
        show_result(gx_test_val, os.path.join(self._output_dir, "test_result.jpg"))
        
    def save(self):
        model_dir = os.path.join(self._checkpoint_dir, 'model')
        self._saver.save(self._sess, os.path.join(model_dir, self._model_name), global_step=self.global_step)
        