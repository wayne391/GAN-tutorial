import os.path
import numpy as np
import tensorflow as tf
from utils.image_io import *
import glob
import shutil


class TrainerOperations(object):
    """Utitlity class for Trainer"""
    def __init__(self, sess, config, name='model'):
        self.sess = sess
        self.name = name
        self.config = config
        self.global_step = None
        self.saver = None

    def init_all(self):
        """Initialize all variables in the scope."""
        tf.global_variables_initializer().run()

    def get_statistics(self):
        """Return model statistics (number of paramaters for each component)."""
        def get_num_parameter(var_list):
            """Given the variable list, return the total number of parameters.
            """
            return int(np.sum([np.product([x.value for x in var.get_shape()])
                               for var in var_list]))

        num_par_g = get_num_parameter(self.model.G.vars)
        num_par_d = get_num_parameter(self.model.D.vars)
        return ("Number of parameters: {}\nNumber of parameters in G: {}\n"
                "Number of parameters in D: {}".format(num_par_g+num_par_d, num_par_g,
                                                       num_par_d))

    def get_global_step_str(self):
        """Return the global step as a string."""
        return str(tf.train.global_step(self.sess, self.global_step))

    def print_statistics(self):
        """Print model statistics (number of paramaters for each component)."""
        print("{:=^80}".format(' Model Statistics '))
        print(self.get_statistics())

    def save_statistics(self, filepath=None):
        """Save model statistics to file. Default to save to the log directory
        given as a global variable."""
        if filepath is None:
            filepath = os.path.join(self.config['dirs']['log'],
                                    'model_statistics.txt')
        with open(filepath, 'w') as f:
            f.write(self.get_statistics())

    def save(self, filepath=None):
        """Save the model to a checkpoint file. Default to save to the log
        directory given as a global variable."""
        if filepath is None:
            filepath = os.path.join(self.config['dirs']['ckpt'],
                                    self.name + '.model')
        print('[*] Saving checkpoint...')
        self.saver.save(self.sess, filepath, self.global_step)

    def load_ckpt(self, filepath):
        """Load the model from the latest checkpoint in a directory."""
        print('[*] Loading checkpoint...')
        self.saver.restore(self.sess, filepath)
        print('Success!!\n\n--\n\n')

    def load(self, checkpoint_dir=None, idx=-1):
        '''List existing checkpoints and load by index'''
        if checkpoint_dir is None:
            checkpoint_dir = self.config['dirs']['ckpt']

        ckpt_list = tf.train.get_checkpoint_state(
            self.config['dirs']['ckpt']).all_model_checkpoint_paths

        for c in ckpt_list:
            print(c)

        ckpt_chosen = ckpt_list[idx]
        print('\n[*] Loading checkpoint...\n')
        print('> you chose:')
        print('> %d, %s' %(idx, ckpt_chosen))
        self.load_ckpt(ckpt_chosen)

    def init_dirs(self):
        for key, path in self.config['dirs'].items():
            if not os.path.exists(path):
                os.makedirs(path)

    def backup_codes(self):
        print('[*] backup codes...')
        copy_dirs = ['.', 'models', 'presets']
        copy_list = []

        for c in copy_dirs:
            tmp_list = glob.glob(os.path.join(c, '*.py'))
            copy_list += tmp_list

        for f in copy_list:
            print('    ->', f)
            shutil.copy2(f, os.path.join(self.config['dirs']['src']))

    def on_save(self, samples, name, path=None):
        path = path if path is not None else self.config['dirs']['sample']
        save_result(samples, grid_size=self.config['sample_grid'],
                            path=path,
                            name=name,
                            denorm_func=self.config['denorm'] if 'denorm' in self.config else None,
                            colormap=self.config['colormap'] if 'colormap' in self.config else None,
                            thres=self.config['thres'] if 'thres' in self.config else None)

    def run_sampler(self, targets, feed_dict, prefix='sample', path=None, is_plot=True):
        print ('*running sampler...')
        samples = self.sess.run(targets, feed_dict=feed_dict)
        path = path if path is not None else self.config['dirs']['sample']

        if is_plot:
            self.on_save(samples, prefix+'_'+str(self.get_global_step_str()), path)

        return samples
