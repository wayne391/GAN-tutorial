from model import Generator, Discriminator
import time
import datetime
import os
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import Saver


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # determine the type of tensor (GPU or CPU)
        self.is_cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor

        # network initialization
        img_shape = (args.channels, args.img_size, args.img_size)
        self.generator = Generator(args, img_shape)
        self.discriminator = Discriminator(img_shape)

        # to GPU
        if self.is_cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        # init dirs
        self.save_dir = os.path.join(args.exp_dir, args.exp_name)
        self.gen_dir = os.path.join(self.save_dir, args.gen_dir)
        self.result_dir = os.path.join(self.save_dir, args.result_dir)
        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # save
        self.saver = Saver(self.save_dir)
        self.save_dict = {
            'generator': self.generator,
            'discriminator': self.discriminator,
        }

        # Info
        print('\n------ Model Info ------')
        print('amount of parameters:', self.network_paras())
        print('Using GPU:', self.is_cuda)
        print('{:=^40}'.format(' Completed '))

    def _compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def network_paras(self):
        # compute only trainable
        g_parameters = filter(lambda p: p.requires_grad, self.generator.parameters())
        d_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        params = sum([np.prod(p.size()) for p in g_parameters]) + sum([np.prod(p.size()) for p in d_parameters])
        return params

    def restore(self):
        self.save_dict = self.saver.load_multiple_model(self.save_dict)

    def train(self, dataset):
        # data
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.args.num_threads)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                       lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                       lr=self.args.lr, betas=(self.args.b1, self.args.b2))

        # ----------------- #
        #  Start Trainning  #
        # ----------------- #

        counter = 0
        previous_time = train_start_time = time.time()

        for epoch in range(self.args.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                # save one patch of real image
                if counter == 0:
                    save_image(imgs.data[:25], os.path.join(
                        self.gen_dir, "real.png"), nrow=5, normalize=True)

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # --------------------- #
                #  Train Discriminator  #
                # --------------------- #

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.args.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = self._compute_gradient_penalty(self.discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.args.lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()
                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % self.args.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                # --------------------  #
                # Monitoring and Saving #
                # --------------------- #

                # print
                if counter and counter % self.args.iter_to_print == 0:
                    cut_time = time.time()
                    interval_time = cut_time - previous_time
                    acc_time = cut_time - train_start_time
                    previous_time = cut_time
                    print(
                        "[Epoch: %d/%d | Batch: %d/%d | Time: %.3f] D loss: %f | G loss: %f | Time: %s" % (
                            epoch,
                            self.args.n_epochs,
                            i,
                            len(dataloader),
                            interval_time,
                            d_loss.item(),
                            g_loss.item(),
                            str(datetime.timedelta(seconds=acc_time)))
                    )

                # save model and image
                if counter and counter % self.args.iter_to_save == 0:
                    save_image(fake_imgs.data[:25], os.path.join(
                        self.gen_dir + "/%d.png" % counter), nrow=5, normalize=True)
                    self.saver.save_multiple_model(self.save_dict)

                counter += 1

        print('{:=^40}'.format(' Finish '))
        runtime = time.time() - train_start_time
        print('training time:', str(datetime.timedelta(seconds=runtime))+'\n\n')

    def gen(self, num=64):
        z = Variable(self.Tensor(np.random.normal(0, 1, (num, self.args.latent_dim))))
        fake_imgs = self.generator(z)
        save_image(fake_imgs.data, os.path.join(
                    self.result_dir, "results.png"), nrow=8, normalize=True)
