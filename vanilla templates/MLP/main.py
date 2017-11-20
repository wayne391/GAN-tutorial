import os
import sys
import shutil
import time
import pprint as pp
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import MLPGAN
from utils import *


flags = tf.app.flags
flags.DEFINE_integer("epoch", 500, "Epoch to train the network (default: 500)")
flags.DEFINE_integer("h1_size", 150, "Number of neurons of 1st hidden layer (default: 150)")
flags.DEFINE_integer("h2_size", 300, "Number of neurons of 2nd hidden layer (default: 300)")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise vector z (default: 100)")
flags.DEFINE_integer("batch_size", 256, "Batch size (default: 256)")
flags.DEFINE_integer("image_size", 784, "Image size (default: 784)")
flags.DEFINE_integer("sample_duration", 5, "Duration of sampling a generated result (default: 5")
flags.DEFINE_float("keep_prob", 0.66, "Value of dropout layer (default: 0.66)")
flags.DEFINE_float("d_lr", 0.0003, "Learning rate of discriminator (default: 0.0003)")
flags.DEFINE_float("g_lr", 0.0001, "Learning rate of generator (default: 0.0001)")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing (default: True)")
flags.DEFINE_boolean("is_restore", False, "True for restoring model, False for not restoring (default: False)")
flags.DEFINE_string("data_dir", "./data", "Output folder path (default: \"./data\")")
flags.DEFINE_string("checkpoint_dir", "./checkpoint-dp", "Output folder path (default: \"./checkpoint\")")
flags.DEFINE_string("output_dir", "./output-dp", "Output folder path (default: \"./output\")")
flags.DEFINE_string("log_dir", "./log-dp", "Output folder path (default: \"./log\")")
flags.DEFINE_string("model_name", "MNIST_MLPGAN.model", "Model name (default: \"MNIST_MLPGAN.model\")")
FLAGS = flags.FLAGS

# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    if FLAGS.is_train:
        if os.path.exists(FLAGS.checkpoint_dir):
            shutil.rmtree(FLAGS.checkpoint_dir, ignore_errors=True)
        if os.path.exists(FLAGS.output_dir):
            shutil.rmtree(FLAGS.output_dir, ignore_errors=True)
        if os.path.exists(FLAGS.log_dir):
            shutil.rmtree(FLAGS.log_dir, ignore_errors=True)
        if os.path.exists(os.path.join(FLAGS.checkpoint_dir, 'model')):
            shutil.rmtree(os.path.join(FLAGS.checkpoint_dir, 'model'), ignore_errors=True)
        
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.output_dir)
    tl.files.exists_or_mkdir(FLAGS.log_dir)
    tl.files.exists_or_mkdir(os.path.join(FLAGS.checkpoint_dir, 'model'))
    
    with tf.Session(config=config) as sess:
        mlpgan = MLPGAN(sess, image_size=FLAGS.image_size,
                       hidden_layer_size=[FLAGS.h1_size, FLAGS.h2_size],
                       keep_prob=FLAGS.keep_prob, d_lr=FLAGS.d_lr, g_lr=FLAGS.g_lr,
                       epoch=FLAGS.epoch, batch_size=FLAGS.batch_size, z_dim=FLAGS.z_dim, 
                       data_dir=FLAGS.data_dir, checkpoint_dir=FLAGS.checkpoint_dir,
                       output_dir=FLAGS.output_dir, log_dir=FLAGS.log_dir, model_name=FLAGS.model_name,
                       sample_duration=FLAGS.sample_duration, is_restore=FLAGS.is_restore)
        mlpgan.train()
        make_gif(mlpgan._img_cache_fixed, os.path.join(FLAGS.output_dir, 'fixed-sample.gif'), true_image=True)
        make_gif(mlpgan._img_cache_random, os.path.join(FLAGS.output_dir, 'random-sample.gif'), true_image=True)

if __name__ == '__main__':
    tf.app.run()