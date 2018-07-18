import os.path
import numpy as np
import tensorflow as tf
from meta import TrainerOperations
import time
import datetime
from utils.image_io import *

class Trainer(TrainerOperations):
    """Class that defines the first-stage (without refiner) model."""
    def __init__(self, sess, model, config):
        super().__init__(sess, config)

        # Initialize dirs
        self.init_dirs()

        # Initialize global_step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Model
        self.model = model

        # Optimizer
        self.g_step = self.model.g_opt.minimize(self.model.g_loss, self.global_step, var_list=self.model.g_vars)
        self.d_step = self.model.d_opt.minimize(self.model.d_loss, self.global_step, var_list=self.model.d_vars)

        # Initialize tf vars
        self.init_all()

        # Saver
        self.saver = tf.train.Saver()

    def train(self, x_train):
        """Train the model."""
        # Print and save model information
        self.backup_codes()
        self.print_statistics()
        self.save_statistics()

        # Check settings
        print ('\n{:-^20}'.format('check setting'))
        print('exp_name:', self.config['exp_name'])
        print(' dataset:', self.config['path_dataset'], x_train.shape)

        # Initialize samples
            # Plot samples of training data
        x_sample = x_train[np.random.choice(
            len(x_train), self.config['num_sample'], False)]
        self.on_save(x_sample, 'train')

            # Plot samples along training
        x_sample = x_train[np.random.choice(
            len(x_train), self.config['num_sample'], False)]
        z_sample = np.random.normal(size=(self.config['num_sample'], self.config['z_dim']))
        feed_dict_sample = {self.model.x:x_sample, self.model.z: z_sample, self.model.is_training:False}

        # Initialize counter
        counter = 0
        time_accumulated = 0.0

        # Open log files and write headers
        log_step = open(os.path.join(self.config['dirs']['log'], 'step.log'), 'w')
        log_batch = open(os.path.join(self.config['dirs']['log'], 'batch.log'), 'w')
        log_epoch = open(os.path.join(self.config['dirs']['log'], 'epoch.log'), 'w')
        log_step.write('# epoch, step, negative_critic_loss\n')
        log_batch.write('# epoch, batch, time, negative_critic_loss, g_loss\n')
        log_epoch.write('# epoch, time, negative_critic_loss, g_loss\n')

        # number of Batch
        num_batch = len(x_train) // self.config['batch_size']

        # epoch iteration
        print('{:=^80}'.format(' Training Start '))
        time_train_start = time.time()
        for epoch in range(self.config['num_epoch']):

            # online shuffle
            shffule_idx = np.random.permutation(len(x_train))

            # batch iteration
            for bidx in range(num_batch):
                batch_start_time = time.time()
                batch_idx = shffule_idx[bidx*self.config['batch_size']: (bidx+1)*self.config['batch_size']]

                z_batch = np.random.normal(size=(self.config['batch_size'], self.config['z_dim']))
                x_batch = x_train[batch_idx]
                feed_dict_batch = {self.model.x: x_batch,
                                    self.model.z: z_batch,
                                    self.model.is_training:True}

                # num of critics
                num_critics = self.config['update'](counter)

                # update networks
                for _ in range(num_critics):
                    _, d_loss = self.sess.run([self.d_step, self.model.d_loss],
                                              feed_dict_batch)
                    log_step.write("{}, {:14.6f}\n".format(
                        self.get_global_step_str(), -d_loss
                    ))

                _, d_loss, g_loss = self.sess.run([self.g_step, self.model.d_loss, self.model.g_loss], feed_dict_batch)

                log_step.write("{}, {:14.6f}\n".format(
                    self.get_global_step_str(), -d_loss
                ))

                time_batch = time.time() - batch_start_time
                time_accumulated = time.time() - time_train_start

                # Print iteration summary
                if self.config['verbose']:
                    if bidx < 1:
                        print("gpu | epoch |   batch   |  time  |       time      |    - D_loss    |"
                              "     G_loss    |     exp")
                    print("{:3s} | {:2d}  | {:4d}/{:4d} | {:6.2f} | {:15s} | {:14.6f} | "
                          "{:14.6f}|   {:30s}".format(self.config['gpu'], epoch, bidx, num_batch, time_batch,  str(datetime.timedelta(seconds=time_accumulated)),
                                            -d_loss, g_loss, self.config['exp_name']))

                log_batch.write("{:d}, {:d}, {:f}, {:f}, {:f}\n".format(
                    epoch, bidx, time_batch, -d_loss, g_loss
                ))

                # run sampler
                if self.config['sample_along_training'] is not None and self.config['sample_along_training'](counter):
                    self.run_sampler(self.model.x_hat, feed_dict_sample)

                counter += 1

            # save checkpoints
            self.save()

        print('{:=^80}'.format(' Training End '))

    def gen(self, num=256, filepath=None, batch_size=None, path=None, is_plot=True):
        batch_size = batch_size if batch_size is not None else self.config['batch_size']
        path = path if path is not None else os.path.join(self.config['exp_name'], 'gen')

        if not os.path.exists(path):
            os.makedirs(path)

        num_batch = num // batch_size

        result_list = []
        for bidx in range(num_batch):
            print('%d/%d'%(bidx, num_batch))
            z_batch = np.random.normal(size=(batch_size, self.config['z_dim']))
            feed_dict_batch = {self.model.z: z_batch,
                               self.model.is_training:False}

            result = self.run_sampler(self.model.x_hat, feed_dict_batch, 'gen_'+str(bidx), path, is_plot=is_plot)
            result_list.append(result)

        print('Done!!')
        ouput = np.concatenate(result_list, axis=0)
        print(ouput.shape)
        np.save(os.path.join(path, 'gen,npy'), result)



