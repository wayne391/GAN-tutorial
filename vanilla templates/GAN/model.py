from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import *
from utils import *
import time
import numpy as np
import os
import math

down_size = lambda size, stride: int(math.ceil(float(size) / float(stride)))

class GAN(object):
    def __init__(self, sess, dataset_dir=None, checkpoint_dir=None, model_dir=None, sample_dir=None, epoch=500, batch_size=256, z_dim=100, epoch_to_sample=50):

        self.sess = sess
        self.dataset_dir = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_dir = model_dir
        self.sample_dir = sample_dir
        print('[*] Loading Data')
        # self.x_train, _ , self.x_test, _ = read_input(self.dataset_dir)
        self.x_train = self.load_mnist()
        idx = np.arange(len(self.x_train))
        np.random.shuffle(idx)
        self.x_train =self.x_train[idx]
        print('[*] Finished!')
        self.z_dim = z_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.epoch_to_sample = epoch_to_sample
        
        self.beta1 = 0.5
        self.d_learning_rate = 2e-4
        self.g_learning_rate = 1e-3

        ## Batch Normalization
        # D
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # G
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.build_model()
       
        

    def build_model(self):
        
        self.x = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='x')       
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')


        self.G = self.generator(self.z, name='G')
        self.D = self.discriminator(self.x, reuse=False, name='D')
        self.sample = self.generator(self.z, reuse=True, name='G')
        self.D_ = self.discriminator(self.G, reuse=True, name='D')
        
        # self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=tf.ones_like(self.D)*0.9))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_,labels=tf.ones_like(self.D_)*0.1))
        # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_, labels=tf.ones_like(self.D_)*0.9))
        # self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2


        self.d_loss = tf.reduce_mean(-(tf.log(self.D) + tf.log(1. - self.D_)))
        self.g_loss = tf.reduce_mean(-(tf.log(self.D_)))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
 

        print(len(self.d_vars))
        
        

        self.d_optimizer = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optimizer =tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)

        # Summation
        self.z_sum = tf.summary.histogram("z", self.z)
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.saver = tf.train.Saver()
        

    def train(self):
        tf.global_variables_initializer().run()

        numOfbatch = self.x_train.shape[0] // self.batch_size
        
        sample_z = np.random.uniform(-1., 1., size=(self.batch_size, self.z_dim)).astype(np.float32)

        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", graph=self.sess.graph)

        show_result(self.x_train[:self.batch_size], os.path.join(self.sample_dir, 'train.jpg'))

        for epoch in range(self.epoch):
            for idx in range(numOfbatch):
                start_time = time.time()

                x_batch = self.x_train[idx*self.batch_size:(idx+1)*self.batch_size]
                z_batch = np.random.uniform(-1., 1., size=(self.batch_size, self.z_dim)).astype(np.float32)

                _, summary_str_d = self.sess.run([self.d_optimizer, self.d_sum], feed_dict={self.x:x_batch, self.z:z_batch})
                _, summary_str_g = self.sess.run([self.g_optimizer, self.g_sum], feed_dict={self.x:x_batch, self.z:z_batch})

                errD = self.d_loss.eval({self.x:x_batch, self.z: z_batch})
                errG = self.g_loss.eval({self.z: z_batch})
                
                print(("Epoch: %2d/%2d [%4d/%4d] time: %4.4f | g_loss:%.6f d_loss:%.6f") 
                % (epoch+1, self.epoch, idx, numOfbatch-1,time.time() - start_time, errG, errD))
 
           
            if epoch % self.epoch_to_sample == 0:

                samples = self.sess.run(self.sample, feed_dict={self.z:sample_z })
                show_result(samples, os.path.join(self.sample_dir, 'sample_epoch_%s.jpg' % epoch))
                
            self.writer.add_summary(summary_str_d, epoch)
            self.writer.add_summary(summary_str_g, epoch)
            self.save(epoch)  
            


    def discriminator(self, x, reuse=False, name='D'):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = lrelu(conv2d(x, 64, k_h=4, k_w=4,name='d_h0_conv'), name='d_h0_prelu')
            h1 = lrelu(self.d_bn1(conv2d(h0, 128, k_h=4, k_w=4, name='d_h1_conv')), name='d_h1_prelu')
            h1 = tf.reshape(h1, [self.batch_size, -1])    
            h2 = lrelu(self.d_bn2(dense(h1, 1024, name='d_h2_lin')), name='d_h2_prelu')

            h3 =  tf.nn.sigmoid(dense(h2, 1, name='d_h3_lin'))


            return h3

    def generator(self, z, reuse=False, name='G'):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            
            s = 28
            s2 = down_size(s , 2) # 14
            s4 = down_size(s2, 2) # 7
           

            h0 = tf.nn.relu(self.g_bn0(dense(z, 1024, name='g_h0_lin')), name='g_h0_prelu')
            h1 = tf.nn.relu(self.g_bn1(dense(z, 64*2*s4*s4, name='g_h1_lin')), name='g_h1_prelu')
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, 64*2])
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size,s2,s2, 64* 2], k_h=4, k_w=4, name='g_h2')), name='g_h2_prelu')
            h3 = tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, 1], k_h=4, k_w=4, name='g_h3'))


            return h3

    def gen_test(self):
        pass
        
    def save(self, step):
        model_name = "MNIST_DCGAN.model"
        new_checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)

        if not os.path.exists(new_checkpoint_dir):
            os.makedirs(new_checkpoint_dir)

        self.saver.save(self.sess, os.path.join(new_checkpoint_dir, model_name), global_step=step)
    
   
    def load(self, reuse_model_dir=None):
        print(" [*] Reading checkpoints...")
        reuse_checkpoint_dir = os.path.join(self.checkpoint_dir, reuse_model_dir)

        ckpt = tf.train.get_checkpoint_state(reuse_checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(reuse_checkpoint_dir, ckpt_name))
        else:
            print('Load Error!')
            
    def load_mnist(self):

        data_dir = "../../Datasets/MNIST"
        
        fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)

        return X/255.