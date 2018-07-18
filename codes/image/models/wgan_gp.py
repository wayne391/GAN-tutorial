import tensorflow as tf
import importlib


class ModelConfig:
    update = lambda cnt: 100 if (cnt < 25) or (cnt % 500 == 0) else 5
    z_dim =  128
    gtype = 0
    dtype = 0
    beta1 = 0.5

class Net(object):
    def __init__(self, config):
        print ('{:=^80}'.format('Building WGAN-GP'))
        self.config = config

        # import presets
        self.presets = importlib.import_module(self.config['path_preset'])

        # build graph
        self._building_graph()
        self.saver = tf.train.Saver()

    def _building_graph(self):
        with tf.variable_scope('model'):

            # shapes
            self.x_shape = (self.config['batch_size'], self.config['height'],
                                  self.config['width'],
                                   self.config['channel'])

            self.z_shape = (self.config['batch_size'], self.config['z_dim'])

            # placeholders
            self.x = tf.placeholder(tf.float32, self.x_shape, name='x')
            self.z = tf.placeholder(tf.float32, self.z_shape, name='z')
            self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

            # subnetworks
            self.G =  self.presets.Generator(is_bn=True, mtype=self.config['gtype'])
            self.D =  self.presets.Discriminator(is_bn=False, mtype=self.config['dtype'])

            # gan
            self.x_hat = self.G(self.z, reuse=None, is_training=self.is_training)
            self.d_real = self.D(self.x, reuse=None)
            self.d_fake = self.D(self.x_hat, reuse=True)

            # regular loss
            self.d_loss = tf.reduce_mean(self.d_fake) - tf.reduce_mean(self.d_real)
            self.g_loss = -tf.reduce_mean(self.d_fake)

            ## compute gradient panelty
            # reshape data
            re_x = tf.reshape(self.x, [self.config['batch_size'], -1])
            re_x_hat = tf.reshape(self.x_hat, [self.config['batch_size'], -1])

            # sample alpha from uniform
            alpha = tf.random_uniform(
                                shape=[self.config['batch_size'], 1],
                                minval=0.,
                                maxval=1.)

            differences = re_x_hat - re_x
            interpolates = re_x + (alpha*differences)

            # feed interpolate into D
            x_inter = tf.reshape(interpolates, self.x_shape)
            self.D_inter = self.D(x_inter, reuse=True)

            # compute gradients panelty
            gradients = tf.gradients(self.D_inter, [interpolates])[0]
            slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2) * 10

            # final d loss
            self.d_loss += gradient_penalty

            # vars
            self.g_vars = self.G.vars
            self.d_vars = self.D.vars

            # optimization (wgan-gp use ADAM)
            self.g_opt = tf.train.AdamOptimizer(self.config['optimizer']['lr'],
                beta1=self.config['optimizer']['beta1'],
                beta2=self.config['optimizer']['beta2'])

            self.d_opt = tf.train.AdamOptimizer(self.config['optimizer']['lr'],
                beta1=self.config['optimizer']['beta1'],
                beta2=self.config['optimizer']['beta2'])



