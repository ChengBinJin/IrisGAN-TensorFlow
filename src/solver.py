# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import tensorflow as tf

import utils as utils


class Solver(object):
    def __init__(self, model, dataset_name, log_dir):
        self.sess = tf.Session()
        self.model = model
        self.dataset_name = dataset_name
        self.log_dir = log_dir

        self.init_global_variables()

    def init_global_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def sample(self, idx, sample_dir, is_save=True, wsize=4, hsize=3):
        feed = {
            self.model.is_train_mode_tfph: False
        }

        g_samples, real_imgs = self.sess.run([self.model.g_samples, self.model.real_imgs], feed_dict=feed)

        if is_save:
            g_samples = utils.unnormalizeUint8(g_samples)
            real_imgs = utils.unnormalizeUint8(real_imgs)

            num_imgs, h, w = g_samples.shape
            # Create figure with num_imgsx2 sub-plots
            fig, axes = plt.subplots(nrows=num_imgs, ncols=2, figsize=(wsize*2, num_imgs*hsize))
            fig.subplots_adjust(hspace=0.05, wspace=0.05)

            for i, ax in enumerate(axes.flat):
                j = i // 2
                # Plot image
                if np.mod(i, 2) == 0:
                    ax.imshow(g_samples[j], cmap='gray')
                elif np.mod(i, 2) == 1:
                    ax.imshow(real_imgs[j], cmap='gray')

                # Remove ticks from the plt
                ax.set_xticks([])
                ax.set_yticks([])

            # Save figure
            plt.savefig(os.path.join(sample_dir, str(idx).zfill(7) + '.png'), bbox_inches='tight')
            # Close figure
            plt.close(fig)


    def train(self):
        feed = {
            self.model.is_train_mode_tfph: True
        }

        # Update discriminator first
        _, d_loss = self.sess.run([self.model.dis_optim, self.model.dis_loss], feed_dict=feed)

        # Run g_optim twice to make sure that d_loss does not got to zero
        g_loss = 0.
        for i in range(2):
            _, g_loss, g_samples = self.sess.run([self.model.gen_optim, self.model.gen_loss, self.model.g_samples],
                                                 feed_dict=feed)

        return d_loss, g_loss

    def saveAugment(self, wsize=4, hsize=3):
        run_op = [self.model.img_ori, self.model.img_trans, self.model.img_flip, self.model.img_rotate]
        img_ori, img_trans, img_flip, img_rotate = self.sess.run(run_op)

        num_imgs, h, w, _ = img_ori.shape
        # Create figure with num_imgsx4 sub-plots
        fig, axes = plt.subplots(nrows=num_imgs, ncols=4, figsize=(wsize*4, num_imgs*hsize))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        for i, ax in enumerate(axes.flat):
            # Plot image
            j = i // 4
            if np.mod(i, 4) == 0:
                ax.imshow(img_ori[j, :, :, 0].astype(np.uint8), cmap='gray')
                xlabel = "Original"
            elif np.mod(i, 4) == 1:
                ax.imshow(img_trans[j, :, :, 0].astype(np.uint8), cmap='gray')
                xlabel = "Translation"
            elif np.mod(i, 4) == 2:
                ax.imshow(img_flip[j, :, :, 0].astype(np.uint8), cmap='gray')
                xlabel = "Flip"
            else:
                ax.imshow(img_rotate[j, :, :, 0].astype(np.uint8), cmap='gray')
                xlabel = "Rotation"

            # Show the label just for last row
            if j == num_imgs - 1:
                ax.set_xlabel(xlabel, fontsize=16)

            # Remove ticks from the plot
            ax.set_xticks([])
            ax.set_yticks([])

        # Save figure
        plt.savefig(os.path.join(self.log_dir, self.dataset_name + '.png'), bbox_inches='tight')
        # Close figure
        plt.close(fig)

