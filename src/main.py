# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import logging
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset
from model import DCGAN

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'CASIA-Iris-Thousand', 'dataset name, default: CASIA-Iris-Thousand')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for one iteration, default: 80')
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
                  total_iters=round(data.num_images * FLAGS.epoch / FLAGS.batch_size),
                  is_train=FLAGS.is_train,
                  log_dir=log_dir)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    iter_time = 0

    # threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while iter_time < 10:
            imgs = sess.run(model.real_imgs)
            g_loss, d_loss = sess.run([model.gen_loss, model.dis_loss], feed_dict={model.mode_tfph: True})

            print('g_loss: {0:>6.3}, d_loss: {1:>6.3}'.format(g_loss, d_loss))

            # print('imgs shape: {}'.format(imgs.shape))
            #
            # for i in range(imgs.shape[0]):
            #     cv2.imshow('Image', imgs[i].astype(np.uint8))
            #     if cv2.waitKey(0) & 0xFF == 27:
            #         sys.exit(' [!] Esc clicked!')

            iter_time += 1

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stope(e)
    finally:
        # When donw, ask the threads to stop
        coord.request_stop()
        coord.join(threads)




if __name__ == '__main__':
    tf.app.run()
