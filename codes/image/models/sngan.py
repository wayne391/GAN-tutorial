import tensorflow as tf
import importlib

class ModelConfig:
    update = lambda cnt: 1
    z_dim =  128
    gtype = 0
    dtype = 1

class Net(object):
    def __init__(self, config):
        print ('{:=^80}'.format('Building SNGAN'))
        self.config = config

        # import presets
        self.presets = importlib.import_module(self.config['path_preset'])

        # build graph
        self._building_graph()
        self.saver = tf.train.Saver()

    def _building_graph(self):
        with tf.variable_scope('model'):

            # shapes
            self.x_shape = (None, self.config['height'],
                                  self.config['width'],
                                self.config['channel'])

            self.z_shape = (None, self.config['z_dim'])

            # placeholders
            self.x = tf.placeholder(tf.float32, self.x_shape, name='x')
            self.z = tf.placeholder(tf.float32, self.z_shape, name='z')
            self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

            # subnetworks
            self.G = self.presets.Generator(is_bn=True, mtype=self.config['gtype'])
            self.D = self.presets.Discriminator(is_bn=False, mtype=self.config['dtype'])

            # gan
            self.x_hat = self.G(self.z, reuse=None, is_training=self.is_training)
            self.d_real = self.D(self.x, reuse=None, update_collection=None)
            self.d_fake = self.D(self.x_hat, reuse=True, update_collection="NO_OPS")

            # regular loss
            self.d_loss = tf.reduce_mean(tf.nn.softplus(self.d_fake) + tf.nn.softplus(-self.d_real))
            self.g_loss = tf.reduce_mean(tf.nn.softplus(-self.d_fake))

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

