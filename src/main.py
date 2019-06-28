# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import csv
import logging
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset
from model import DCGAN
from solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'CASIA-Iris-Thousand', 'dataset name, default: CASIA-Iris-Thousand')
tf.flags.DEFINE_string('method', 'wgan-gp', 'GAN method [dcgan|wgan-gp], default: WGAN-GP')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size for one iteration, default: 64')
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
    if FLAGS.method.lower() == 'wgan-gp':
        model = WGAN_GP()
    else:
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

    # Initialize saver
    saver = tf.train.Saver(max_to_keep=1)

    if FLAGS.is_train:
        train(solver, data, saver, logger, sample_dir, model_dir, log_dir)
    else:
        test(solver, saver, test_dir, model_dir, log_dir)


def train(solver, data, saver, logger, sample_dir, model_dir, log_dir):
    iter_time = 0
    one_epoch_iters = int(np.ceil(data.num_images / FLAGS.batch_size))
    total_iters = int(np.ceil(FLAGS.epoch * data.num_images / FLAGS.batch_size))

    if FLAGS.load_model is not None:
        flag, iter_time = load_model(saver=saver, solver=solver, logger=logger, model_dir=model_dir, is_train=True)
        logger.info(' [!] Load Success! Iter: {}'.format(iter_time))

    # Tensorboard writer
    tb_writer = tf.summary.FileWriter(logdir=log_dir,
                                      graph_def=solver.sess.graph_def)

    csvWriter = utils.CSVWriter(path=model_dir)

    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=solver.sess, coord=coord)

    try:
        while iter_time <= total_iters:
            # Save training images just in first iteration
            if iter_time == 0:
                solver.saveAugment()

            d_loss, g_loss, summary = solver.train()

            # Write to tensorboard
            tb_writer.add_summary(summary, iter_time)
            tb_writer.flush()

            if (iter_time % FLAGS.print_freq == 0) or (iter_time == total_iters):
                # Write loss information on the csv file
                csvWriter.update(iter_time=iter_time, d_loss=d_loss, g_loss=g_loss)

                msg = "[{0:>7}/{1:>7}] d_loss: {2:>6.3}, g_loss: {3:>6.3}"
                print(msg.format(iter_time, total_iters, d_loss, g_loss))

            # Sampling random images
            if iter_time % FLAGS.sample_freq == 0 or (iter_time == total_iters):
                solver.sample(idx=iter_time, save_dir=sample_dir, is_save=True)

            # Sampling fixed vectors for finishing one epoch
            # if iter_time % one_epoch_iters == 0 or (iter_time == total_iters):
            if iter_time % 500 == 0 or (iter_time == total_iters):
                solver.fixedSample(save_dir=sample_dir, is_save=True)
                save_model(saver, solver, logger, model_dir, iter_time)

            iter_time += 1

        # Close csv file
        csvWriter.close()

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # When donw, ask the threads to stop
        coord.request_stop()
        coord.join(threads)

    # Plot loss information
    plot_loss(log_dir, model_dir)


def test(solver, saver, test_dir, model_dir, log_dir):
    flag, iter_time = load_model(saver=saver, solver=solver, logger=None, model_dir=model_dir, is_train=False)
    if flag is False:
        sys.exit(" [!] Failed to load model: {}!".format(model_dir))

    # Load csv file to plot loss information
    plot_loss(log_dir, model_dir)
    iter_time, total_iters = 0, 30

    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=solver.sess, coord=coord)

    try:
        while iter_time <= total_iters:
            print("Iter: {}".format(iter_time))
            solver.test_sample(idx=iter_time, save_dir=test_dir)
            iter_time += 1

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # When donw, ask the threads to stop
        coord.request_stop()
        coord.join(threads)


def save_model(saver, solver, logger, model_dir, iter_time):
    np.save(os.path.join(model_dir, 'g_samples.npy'), np.asarray(solver.g_samples))
    saver.save(solver.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
    logger.info(' [*] Model saved! Iter: {}'.format(iter_time))


def load_model(saver, solver, logger, model_dir, is_train=False):
    if is_train:
        logger.info(' [*] Reading checkpoint...')
    else:
        print(' [*] Reading checkpoint...')

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(solver.sess,os.path.join(model_dir, ckpt_name))

        meta_graph_path = ckpt.model_checkpoint_path + '.meta'
        iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

        if is_train:
            logger.info(' [!] Load Iter: {}'.format(iter_time))
        else:
            print(' [!] Load Iter: {}'.format(iter_time))

        # Restore solver.g_samples
        g_samples = np.load(os.path.join(model_dir, 'g_samples.npy'))
        solver.epoch_time = g_samples.shape[0]
        for i in range(g_samples.shape[0]):
            solver.g_samples.append(g_samples[i])

        return True, iter_time + 1
    else:
        return False, None


def plot_loss(log_dir, model_dir):
    # read csv file
    file_name = os.path.join(model_dir, 'loss.csv')

    data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            data.append([float(row[1]), float(row[2])])  # Discriminator and Generator

    # Transpose and become to array: (2, N)
    data = np.asarray(np.transpose(data))

    # Set environment
    sns.set()  # Use seaborn library to draw plt
    plt.rcParams['figure.figsize'] = (12.0, 8.0)  # set default size of plots

    # Draw x and y ticks
    x = np.arange(data.shape[1])
    plt.plot(x, data[0, :], color='green', linestyle='solid')   # discriminator loss
    plt.plot(x, data[1, :], color='blue', linestyle='solid')    # generator loss

    # Add legend, label, and title
    plt.legend(['Discriminator Loss', 'Generator Loss'], loc='upper right')
    plt.title('Loss Functions of CASIA Iris Generator Model')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.savefig(os.path.join(log_dir, 'Loss.png'), bbox_inches='tight', dpi=600)

if __name__ == '__main__':
    tf.app.run()
