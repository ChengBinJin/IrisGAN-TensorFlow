# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import tensorflow as tf
from datetime import datetime

import utils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'CASIA-Iris-Thousand', 'dataset name, default: CASIA-Iris-Thousand')
tf.flags.DEFINE_integer('batch_size', 80, 'batch size for one iteration, default: 80')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of the random vector, default: 100')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for optimizer, default: 0.0001')
tf.flags.DEFINE_float('epoch', 120, 'number of epochs for training, default: 120')
tf.flags.DEFINE_float('print_freq', 100, 'print frequence for loss information, default: 50')
tf.flags.DEFINE_integer('sample_batch', 16, 'sample batch size, default: 16')
tf.flags.DEFINE_float('sample_freq', 500, 'sample frequence for checking quality of the generated images, default: 500')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190611-1516), default: None')

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)

def init_logger(log_dir, is_train=True):
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'main.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if is_train:
        logger.info('gpu_index: \t\t{}'.format(FLAGS.gpu_index))
        logger.info('dataset: \t\t{}'.format(FLAGS.dataset))
        logger.info('batch_size: \t\t{}'.format(FLAGS.batch_size))
        logger.info('z_dim: \t\t{}'.format(FLAGS.z_dim))
        logger.info('is_train: \t\t{}'.format(FLAGS.is_train))
        logger.info('learning_rate: \t{}'.format(FLAGS.learning_rate))
        logger.info('epoch: \t\t{}'.format(FLAGS.epoch))
        logger.info('print_freq: \t\t{}'.format(FLAGS.print_freq))
        logger.info('sample_batch: \t\t{}'.format(FLAGS.sample_batch))
        logger.info('sample_freq: \t\t{}'.format(FLAGS.sample_freq))
        logger.info('load_model: \t\t{}'.format(FLAGS.load_model))
    else:
        print('-- gpu_index: \t\t{}'.format(FLAGS.gpu_index))
        print('-- dataset: \t\t{}'.format(FLAGS.dataset))
        print('-- batch_size: \t\t{}'.format(FLAGS.batch_size))
        print('-- z_dim: \t\t{}'.format(FLAGS.z_dim))
        print('-- is_train: \t\t{}'.format(FLAGS.is_train))
        print('-- learning_rate: \t\t{}'.format(FLAGS.learning_rate))
        print('-- epoch: \t\t{}'.format(FLAGS.epoch))
        print('-- print_freq: \t\t{}'.format(FLAGS.print_freq))
        print('-- sample_batch: \t\t{}'.format(FLAGS.sample_batch))
        print('-- sample_freq: \t\t{}'.format(FLAGS.sample_freq))
        print('-- load_model: \t\t{}'.format(FLAGS.load_model))


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir, sample_dir, test_dir = utils.make_folders(is_train=FLAGS.is_train, cur_time=cur_time)
    init_logger(log_dir=log_dir, is_train=FLAGS.is_train)

    # Initialize dataset

if __name__ == '__main__':
    tf.app.run()
