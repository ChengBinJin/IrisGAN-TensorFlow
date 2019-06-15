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


class Solver(object):
    def __init__(self, sess, model, dataset_name, log_dir):
        self.sess = sess
        self.model = model
        self.dataset_name = dataset_name
        self.log_dir = log_dir

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

