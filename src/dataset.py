# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys

import utils as utils

class CASIAIrisThousand(object):
    def __init__(self, name, is_train, log_dir):
        self.name = name
        self.num_persons = 1000
        self.num_images = 20000
        self.image_shape = (480, 640, 1)

        # Tfrecord path
        self.data_path = '../../Data/CASIA-IrisV4/CASIA-Iris-Thousand.tfrecords'

        if is_train:
            self.logger, self.file_handler, self.stream_handler = utils.init_logger(log_dir=log_dir,
                                                                                    is_train=is_train,
                                                                                    name='dataset')

            self.logger.info('Name: {}'.format(self.name))
            self.logger.info('Number of persons: {}'.format(self.num_persons))
            self.logger.info('Number of images: {}'.format(self.num_images))
            self.logger.info('Image shape: {}'.format(self.image_shape))

    def __call__(self):
        if not os.path.isfile(self.data_path):
            sys.exit(' [!] The tfrecord file {} is not found...'.format(self.data_path))
        return self.data_path


def Dataset(name, is_train, log_dir):
    if name == 'CASIA-Iris-Thousand':
        return CASIAIrisThousand(name, is_train, log_dir)
    else:
        raise NotImplementedError
