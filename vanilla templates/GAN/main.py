from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import GAN
import tensorflow as tf
import pprint
import os

# GPU assignment
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

pp = pprint.PrettyPrinter()
flags = tf.app.flags

flags.DEFINE_string("dataset_dir", "../../Datasets/MNIST", "Directory to dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints")
flags.DEFINE_string("model_dir", "try1", "to save/reuse model")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("epoch",20, "Number of Epochs")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("z_dim", 74, "noise dim")
flags.DEFINE_integer("epoch_to_sample", 1, "epochs to smaple")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        
        model = GAN(sess, 
            dataset_dir=FLAGS.dataset_dir,
            checkpoint_dir=FLAGS.checkpoint_dir,
            model_dir=FLAGS.model_dir,
            sample_dir=FLAGS.sample_dir,
            epoch=FLAGS.epoch,
            batch_size=FLAGS.batch_size,
            z_dim=FLAGS.z_dim,
            epoch_to_sample=FLAGS.epoch_to_sample)

        if FLAGS.is_train:
            model.train()
        else:
            pass

if __name__ == '__main__':
    tf.app.run()