# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset

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


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir, sample_dir, test_dir = utils.make_folders(is_train=FLAGS.is_train, cur_time=cur_time)

    # Logger
    logger, _, _ = utils.init_logger(log_dir=log_dir, is_train=FLAGS.is_train, name='main')
    utils.print_main_parameters(logger, flags=FLAGS, is_train=FLAGS.is_train)

    # Initialize dataset
    data = Dataset(name=FLAGS.dataset, is_train=FLAGS.is_train, log_dir=log_dir)

    # Initialize session
    # sess = tf.Session()

    # Initialize model
    model = Model()


if __name__ == '__main__':
    tf.app.run()
