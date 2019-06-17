# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import logging

import utils as utils


class CASIAIrisThousand(object):
    def __init__(self, name='Data', is_train=True, resize_factor=0.25, log_dir=None):
        self.name = name
        self.num_persons = 1000
        self.num_images = 20000
        self.image_shape = (int(480 * resize_factor), int(640 * resize_factor), 1)

        # Tfrecord path
        self.data_path = '../../Data/CASIA-IRisV4/CASIA-Iris-Thousand.tfrecords'

        if is_train:
            self.logger = logging.getLogger(__name__)  # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, log_dir=log_dir, is_train=is_train, name='dataset')

            self.logger.info('Name: \t\t\t{}'.format(self.name))
            self.logger.info('Number of persons: \t{}'.format(self.num_persons))
            self.logger.info('Number of images: \t{}'.format(self.num_images))
            self.logger.info('Image shape: \t\t{}'.format(self.image_shape))

    def __call__(self):
        if not os.path.isfile(self.data_path):
            sys.exit(' [!] The tfrecord file {} is not found...'.format(self.data_path))
        return self.data_path


def Dataset(name, is_train, resized_factor, log_dir):
    if name == 'CASIA-Iris-Thousand':
        return CASIAIrisThousand(name=name, is_train=is_train, resize_factor=resized_factor, log_dir=log_dir)
    else:
        raise NotImplementedError
