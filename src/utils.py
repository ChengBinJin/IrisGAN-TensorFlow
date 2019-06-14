# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging


def make_folders(is_train=True, cur_time=None):
    model_dir = os.path.join('model', '{}'.format(cur_time))
    log_dir = os.path.join('log', '{}'.format(cur_time))
    sample_dir = os.path.join('sample', '{}'.format(cur_time))
    test_dir = None

    if is_train:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)
    else:
        test_dir = os.path.join('test', '{}'.format(cur_time))

        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    return model_dir, log_dir, sample_dir, test_dir


def init_logger(logger, log_dir, name, is_train):
    file_handler, stream_handler = None, None
    if is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

        # file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger, file_handler, stream_handler


def release_handles(logger, file_handler, stream_handler):
    file_handler.close()
    stream_handler.close()
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)


def print_main_parameters(logger, flags, is_train=True):
    if is_train:
        logger.info('gpu_index: \t\t{}'.format(flags.gpu_index))
        logger.info('dataset: \t\t{}'.format(flags.dataset))
        logger.info('batch_size: \t\t{}'.format(flags.batch_size))
        logger.info('z_dim: \t\t{}'.format(flags.z_dim))
        logger.info('is_train: \t\t{}'.format(flags.is_train))
        logger.info('learning_rate: \t{}'.format(flags.learning_rate))
        logger.info('epoch: \t\t{}'.format(flags.epoch))
        logger.info('print_freq: \t\t{}'.format(flags.print_freq))
        logger.info('sample_batch: \t\t{}'.format(flags.sample_batch))
        logger.info('sample_freq: \t\t{}'.format(flags.sample_freq))
        logger.info('load_model: \t\t{}'.format(flags.load_model))
    else:
        print('-- gpu_index: \t\t{}'.format(flags.gpu_index))
        print('-- dataset: \t\t{}'.format(flags.dataset))
        print('-- batch_size: \t\t{}'.format(flags.batch_size))
        print('-- z_dim: \t\t{}'.format(flags.z_dim))
        print('-- is_train: \t\t{}'.format(flags.is_train))
        print('-- learning_rate: \t\t{}'.format(flags.learning_rate))
        print('-- epoch: \t\t{}'.format(flags.epoch))
        print('-- print_freq: \t\t{}'.format(flags.print_freq))
        print('-- sample_batch: \t\t{}'.format(flags.sample_batch))
        print('-- sample_freq: \t\t{}'.format(flags.sample_freq))
        print('-- load_model: \t\t{}'.format(flags.load_model))
