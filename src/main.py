# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
# import sys
import logging
# import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset
from model import DCGAN
from solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'CASIA-Iris-Thousand', 'dataset name, default: CASIA-Iris-Thousand')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for one iteration, default: 80')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of the random vector, default: 100')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for optimizer, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('epoch', 1, 'number of epochs for training, default: 120')
tf.flags.DEFINE_float('print_freq', 100, 'print frequence for loss information, default: 50')
tf.flags.DEFINE_integer('sample_batch', 16, 'sample batch size, default: 16')
tf.flags.DEFINE_float('sample_freq', 100, 'sample frequence for checking quality of the generated images, default: 500')
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
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, log_dir=log_dir, is_train=FLAGS.is_train, name='main')
    utils.print_main_parameters(logger, flags=FLAGS, is_train=FLAGS.is_train)

    # Initialize dataset
    data = Dataset(name=FLAGS.dataset, is_train=FLAGS.is_train, resized_factor=0.25, log_dir=log_dir)

    # Initialize model
    model = DCGAN(image_shape=data.image_shape,
                  data_path=data(),
                  batch_size=FLAGS.batch_size,
                  z_dim=FLAGS.z_dim,
                  lr=FLAGS.learning_rate,
                  beta1=FLAGS.beta1,
                  total_iters=int(np.ceil(FLAGS.epoch * data.num_images / FLAGS.batch_size)),
                  is_train=FLAGS.is_train,
                  log_dir=log_dir)

    # Intialize solver
    solver = Solver(model=model,
                    dataset_name=data.name,
                    batch_size=FLAGS.batch_size,
                    z_dim=FLAGS.z_dim,
                    log_dir=log_dir)

    if FLAGS.is_train:
        train(solver, data, sample_dir, log_dir)
    else:
        test(solver)


def train(solver, data, sample_dir, log_dir):
    iter_time = 0
    one_epoch_iters = int(np.ceil(data.num_images / FLAGS.batch_size))
    total_iters = int(np.ceil(FLAGS.epoch * data.num_images / FLAGS.batch_size))

    # Tensorboard writer
    tb_writer = tf.summary.FileWriter(log_dir, graph_def=solver.sess.graph_def)

    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=solver.sess, coord=coord)

    try:
        while iter_time < total_iters:
            # Save training images just in first iteration
            if iter_time == 0:
                solver.saveAugment()

            d_loss, g_loss, summary = solver.train()

            # Write to tensorboard
            tb_writer.add_summary(summary, iter_time)
            tb_writer.flush()

            if iter_time % FLAGS.print_freq == 0:
                msg = "[{0:>7}/{1:>7}] d_loss: {2:>6.3}, g_loss: {3:>6.3}"
                print(msg.format(iter_time, total_iters, d_loss, g_loss))

            # Sampling random images
            if iter_time % FLAGS.sample_freq == 0:
                solver.sample(idx=iter_time, sample_dir=sample_dir, is_save=True)

            # Sampling fixed vectors
            if iter_time % one_epoch_iters == 0:
                solver.fixedSample(sample_dir=sample_dir, is_save=True)

            iter_time += 1

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # When donw, ask the threads to stop
        coord.request_stop()
        coord.join(threads)


def test(solver):
    print("Hello test!")



if __name__ == '__main__':
    tf.app.run()
