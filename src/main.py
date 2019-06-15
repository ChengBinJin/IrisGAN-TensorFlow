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
from solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'CASIA-Iris-Thousand', 'dataset name, default: CASIA-Iris-Thousand')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for one iteration, default: 80')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of the random vector, default: 100')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for optimizer, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
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

    # Initialize session
    sess = tf.Session()

    # Initialize dataset
    data = Dataset(name=FLAGS.dataset, is_train=FLAGS.is_train, resized_factor=0.25, log_dir=log_dir)

    # Initialize model
    model = DCGAN(image_shape=data.image_shape,
                  data_path=data(),
                  batch_size=FLAGS.batch_size,
                  z_dim=FLAGS.z_dim,
                  lr=FLAGS.learning_rate,
                  beta1=FLAGS.beta1,
                  total_iters=round(data.num_images * FLAGS.epoch / FLAGS.batch_size),
                  is_train=FLAGS.is_train,
                  log_dir=log_dir)

    # Intialize solver
    solver = Solver(sess=sess,
                    model=model,
                    dataset_name=data.name,
                    log_dir=log_dir)

    sess.run(tf.global_variables_initializer())
    iter_time = 0

    # threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while iter_time < 10:
            if iter_time == 0:
                solver.saveAugment()

            # img_ori, img_trans, img_flip, img_rotate = sess.run(
            #     [model.img_ori, model.img_trans, model.img_flip, model.img_rotate])
            #
            # # g_loss, d_loss = sess.run([model.gen_loss, model.dis_loss], feed_dict={model.mode_tfph: True})
            #
            # # print('g_loss: {0:>6.3}, d_loss: {1:>6.3}'.format(g_loss, d_loss))
            #
            # print('imgs shape: {}'.format(img_ori.shape))
            # num_img, h, w, c = img_ori.shape
            #
            # canvas = np.zeros((num_img * h, num_img * w), dtype=np.uint8)
            # for i in range(num_img):
            #     canvas[i*h:(i+1)*h, 0:w] = img_ori[i, :, :, 0].astype(np.uint8)
            #     canvas[i*h:(i+1)*h, w:2*w] = img_trans[i, :, :, 0].astype(np.uint8)
            #     canvas[i*h:(i+1)*h, 2*w:3*w] = img_flip[i, :, :, 0].astype(np.uint8)
            #     canvas[i*h:(i+1)*h, 3*w:4*w] = img_rotate[i, :, :, 0].astype(np.uint8)
            #
            # cv2.imshow('Image', canvas)
            # if cv2.waitKey(0) & 0xFF == 27:
            #     sys.exit(' [!] Esc clicked!')

            iter_time += 1

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # When donw, ask the threads to stop
        coord.request_stop()
        coord.join(threads)




if __name__ == '__main__':
    tf.app.run()
