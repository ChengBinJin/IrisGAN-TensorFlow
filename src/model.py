# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import logging
import tensorflow as tf

import tensorflow_utils as tf_utils
import utils as utils
from reader import Reader


class WGAN_GP(object):
    def __init__(self, image_shape, data_path, batch_size=64, z_dim=100, lr=2e-4, beta1=0.999, total_iters=2e5,
                 is_train=True, log_dir=None, name='wgan-gp'):
        self.image_shape = image_shape
        self.data_path = data_path
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.gen_dims = [512, 512, 256, 128, 64, 64, 3]
        self.dis_dims = [64, 128, 256, 512, 512, 512, 1]
        self.lr = lr
        self.beta1 = beta1
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.is_train = is_train
        self.log_dir = log_dir
        self.name = name

        self.g_lr_tb, self.d_lr_tb = None, None

        self.logger = logging.getLogger(__name__)   #logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=self.log_dir, is_train=self.is_train, name=self.name)
        self.gen_ops, self.dis_ops = [], []

        self._build_net()
        self._initTensorBoard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _build_net(self):
        print()

    def _initTensorBoard(self):
        dis_loss = tf.summary.scalar('Loss/dis_loss', self.dis_loss)
        gen_loss = tf.summary.scalar('Loss/gen_loss', self.gen_loss)
        dis_lr = tf.summary.scalar('Learning_rate/dis_lr', self.dis_optimizer_obj.learning_rate)
        gen_lr = tf.summary.scalar('Learning_rate/gen_lr', self.gen_optimizer_obj.learning_rate)
        self.summary_op = tf.summary.merge(inputs=[dis_loss, gen_loss, dis_lr, gen_lr])


class DCGAN(object):
    def __init__(self, image_shape, data_path, batch_size=64, z_dim=100, lr=2e-4, beta1=0.999, total_iters=2e5,
                 is_train=True, log_dir=None, name='dcgan'):
        self.image_shape = image_shape
        self.data_path = data_path
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.gen_dims = [1024, 512, 256, 128, 64, 1]
        self.dis_dims = [64, 128, 256, 512, 1024, 1]
        self.lr = lr
        self.beta1 = beta1
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.is_train = is_train
        self.log_dir = log_dir
        self.name = name

        self.g_lr_tb, self.d_lr_tb = None, None

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=self.log_dir, is_train=self.is_train, name=self.name)
        self.gen_ops, self.dis_ops = [], []

        self._build_net()
        self._initTensorBoard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _build_net(self):
        self.z_vector_tfph = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='z_tfph')
        self.is_train_mode_tfph = tf.placeholder(dtype=tf.bool, name='mode_tfph')

        # Initialize generator and discriminator
        self.gen = Generator(name='gen',
                             dims=self.gen_dims,
                             norm='batch',
                             _ops=self.gen_ops,
                             logger=self.logger)
        self.dis = Discriminator(name='dis',
                                 dims=self.dis_dims,
                                 norm='batch',
                                 _ops=self.dis_ops,
                                 logger=self.logger)

        reader = Reader(tfrecords_file=self.data_path,
                        image_shape=self.image_shape,
                        batch_size=self.batch_size,
                        is_train=self.is_train)

        # Data reader
        self.img_ori, self.img_trans, self.img_flip, self.img_rotate, self.img_names = reader.feed()
        self.real_imgs = tf.identity(input=self.normalize(self.img_rotate))

        # Generator and discriminator loss
        self.g_samples = self.gen(x=self.randomVector(), is_train=self.is_train_mode_tfph)
        self.gen_loss = self.generatorLoss(dis_obj=self.dis, fake_img=self.g_samples)
        self.dis_loss = self.discriminatorLoss(dis_obj=self.dis, real_img=self.real_imgs, fake_img=self.g_samples)

        # Optimizers
        self.gen_optimizer_obj = Optimizer(start_learning_rate=self.lr,
                                           start_decay_step=self.start_decay_step,
                                           decay_steps=self.decay_steps,
                                           beta1=self.beta1,
                                           name='Gen_Adam',
                                           is_twice=True)
        gen_op = self.gen_optimizer_obj(loss=self.gen_loss, var_list=self.gen.variables)
        gen_ops = [gen_op] + self.gen_ops
        self.gen_optim = tf.group(*gen_ops)

        self.dis_optimizer_obj = Optimizer(start_learning_rate=self.lr,
                                           start_decay_step=self.start_decay_step,
                                           decay_steps=self.decay_steps,
                                           beta1=self.beta1,
                                           name='Dis_Adam',
                                           is_twice=False)
        dis_op = self.dis_optimizer_obj(loss=self.dis_loss, var_list=self.dis.variables)
        dis_ops = [dis_op] + self.dis_ops
        self.dis_optim = tf.group(*dis_ops)

    def _initTensorBoard(self):
        dis_loss = tf.summary.scalar('Loss/dis_loss', self.dis_loss)
        gen_loss = tf.summary.scalar('Loss/gen_loss', self.gen_loss)
        dis_lr = tf.summary.scalar('Learning_rate/dis_lr', self.dis_optimizer_obj.learning_rate)
        gen_lr = tf.summary.scalar('Learning_rate/gen_lr', self.gen_optimizer_obj.learning_rate)
        self.summary_op = tf.summary.merge(inputs=[dis_loss, gen_loss, dis_lr, gen_lr])

    def randomVector(self):
        random_vector = tf.random.normal(shape=(self.batch_size, self.z_dim), name='random_vector')
        return random_vector

    @staticmethod
    def generatorLoss(dis_obj, fake_img):
        d_logit_fake = dis_obj(fake_img)
        loss = tf.math.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logit_fake, labels=tf.ones_like(tensor=d_logit_fake)))

        return loss

    @staticmethod
    def discriminatorLoss(dis_obj, real_img, fake_img):
        d_logit_real = dis_obj(real_img)
        d_logit_fake = dis_obj(fake_img)

        error_real = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logit_real, labels=tf.ones_like(tensor=d_logit_real)))
        error_fake = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logit_fake, labels=tf.zeros_like(tensor=d_logit_fake)))

        loss = 0.5 * (error_real + error_fake)
        return loss

    @staticmethod
    def normalize(data):
        return data / 127.5 - 1.0


class Optimizer(object):
    def __init__(self, start_learning_rate, start_decay_step, decay_steps, beta1=0.9, name=None, is_twice=False):
        self.start_learning_rate = start_learning_rate
        self.end_leanring_rate = 0.

        if is_twice is True:
            self.start_decay_step = start_decay_step * 2
            self.decay_steps = decay_steps * 2
        else:
            self.start_decay_step = start_decay_step
            self.decay_steps = decay_steps

        self.beta1 = beta1
        self.name = name
        self.learning_rate = None

    def __call__(self, loss, var_list):
        with tf.variable_scope(self.name):
            global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
            self.learning_rate = tf.where(condition=tf.math.greater_equal(x=global_step, y=self.start_decay_step),
                                     x=tf.train.polynomial_decay(learning_rate=self.start_learning_rate,
                                                                 global_step=(global_step - self.start_decay_step),
                                                                 decay_steps=self.decay_steps,
                                                                 end_learning_rate=self.end_leanring_rate,
                                                                 power=1.0),
                                     y=self.start_learning_rate)

            learn_step = tf.train.AdamOptimizer(
                self.learning_rate, beta1=self.beta1).minimize(loss, global_step=global_step, var_list=var_list)

        return learn_step


class ResnetDiscriminator(object):
    def __init__(self, name, dims, norm='batch', _ops=None, logger=None):
        self.name = name


class Discriminator(object):
    def __init__(self, name, dims, norm='batch', _ops=None, logger=None):
        self.name = name
        self.dims = dims
        self.norm = norm
        self._ops = _ops
        self.logger = logger
        self.reuse = False

    def __call__(self, x, is_train=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, 120, 160, 1) -> (N, 60, 80, 64)
            h0_conv = tf_utils.conv2d(x, output_dim=self.dims[0], initializer='he', name='h0_conv',
                                      logger=self.logger if is_train is True else None)
            h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu', logger=self.logger if is_train is True else None)

            # (N, 60, 80, 64) -> (N, 30, 40, 128)
            h1_conv = tf_utils.conv2d(h0_lrelu, output_dim=self.dims[1], initializer='he', name='h1_conv',
                                      logger=self.logger if is_train is True else None)
            h1_norm = tf_utils.norm(h1_conv, name='h1_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h1_lrelu = tf_utils.lrelu(h1_norm, name='h1_lrelu', logger=self.logger if is_train is True else None)

            # (N, 30, 40, 128) -> (N, 15, 20, 256)
            h2_conv = tf_utils.conv2d(h1_lrelu, output_dim=self.dims[2], initializer='he', name='h2_conv',
                                      logger=self.logger if is_train is True else None)
            h2_norm = tf_utils.norm(h2_conv, name='h2_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h2_lrelu = tf_utils.lrelu(h2_norm, name='h2_lrelu', logger=self.logger if is_train is True else None)

            # (N, 15, 20, 256) -> (N, 8, 10, 512)
            h3_conv = tf_utils.conv2d(h2_lrelu, output_dim=self.dims[3], initializer='he', name='h3_conv',
                                      logger=self.logger if is_train is True else None)
            h3_norm = tf_utils.norm(h3_conv, name='h3_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h3_lrelu = tf_utils.lrelu(h3_norm, name='h3_lrelu', logger=self.logger if is_train is True else None)

            # (N, 8, 10, 512) -> (N, 4, 5, 1024)
            h4_conv = tf_utils.conv2d(h3_lrelu, output_dim=self.dims[4], initializer='he', name='h4_conv',
                                      logger=self.logger if is_train is True else None)
            h4_norm = tf_utils.norm(h4_conv, name='h4_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h4_lrelu = tf_utils.lrelu(h4_norm, name='h4_lrelu', logger=self.logger if is_train is True else None)
            # (N, 4, 5, 1024) -> (N, 4*5*1024)
            h4_flatten = tf_utils.flatten(h4_lrelu, name='h4_flatten', logger=self.logger if is_train is True else None)

            # (N, 4*5*1024) -> (N, 1)
            output = tf_utils.linear(h4_flatten, output_size=self.dims[5], initializer='he', name='output',
                                     logger=self.logger if is_train is True else None)

            # Set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class ResnetGenerator(object):
    def __init__(self, name, dims, norm='batch', _ops=None, logger=None):
        self.name = name
        self.dims = dims
        self.norm = norm
        self._ops = _ops
        self.logger = logger
        self.reuse = False

    def __call__(self, x, is_train=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, 100) -> (N, 4, 5, 512)
            h0_linear = tf_utils.linear(x, 4*5*self.dims[0], name='h0_linear', initializer='He',
                                        logger=self.logger if is_train is True else None)
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 5, self.dims[0]])

            # (N, 4, 5, 512) -> (N, 8, 10, 512)
            resblock_1 = tf_utils.res_block_v2(x=h0_reshape, k=self.dims[1], filter_size=3, _ops=self._ops,
                                               norm_='batch', resample='up', name='res_block_1',
                                               logger=self.logger if is_train is True else None)

            # (N, 8, 10, 512) -> (N, 16, 20, 256)
            resblock_2 = tf_utils.res_block_v2(x=resblock_1, k=self.dims[2], filter_size=3, _ops=self._ops,
                                               norm_='batch', resample='up', name='res_block_2',
                                               logger=self.logger if is_train is True else None)

            # (N, 16, 20, 256) -> (N, 15, 20, 256)
            resblock_2_split, _ = tf.split(resblock_2, [15, 1], axis=1, name='resblock_2_split')
            tf_utils.print_activations(resblock_2_split, logger=self.logger if is_train is True else None)

            # (N, 15, 20, 256) -> (N, 30, 40, 128)
            resblock_3 = tf_utils.res_block_v2(x=resblock_2_split, k=self.dims[3], filter_size=3, _ops=self._ops,
                                               norm_='batch', resample='up', name='res_block_3',
                                               logger=self.logger if is_train is True else None)

            # (N, 30, 40, 128) -> (N, 60, 80, 64)
            resblock_4 = tf_utils.res_block_v2(x=resblock_3, k=self.dims[4], filter_size=3, _ops=self._ops,
                                               norm_='batch', resample='up', name='res_block_4',
                                               logger=self.logger if is_train is True else None)

            # (N, 60, 80, 64) -> (N, 120, 160, 64)
            resblock_5 = tf_utils.res_block_v2(x=resblock_4, k=self.dims[5], filter_size=3, _ops=self._ops,
                                               norm_='batch', resample='up', name='res_block_5',
                                               logger=self.logger if is_train is True else None)

            norm_5 = tf_utils.norm(resblock_5, name='norm_5', _type='batch', _ops=self._ops, is_train=is_train,
                                   logger=self.logger if is_train is True else None)

            relu_5 = tf_utils.relu(norm_5, name='relu_5', logger=self.logger if is_train is True else None)

            # (N, 120, 160, 64) -> (N, 120, 160, 3)
            conv_6 = tf_utils.conv2d(relu_5, output_dim=self.dims[6], k_h=3, k_w=3, d_h=1, d_w=1, name='conv_6',
                                     logger=self.logger if is_train is True else None)

            output = tf_utils.tanh(conv_6, name='output', logger=self.logger if is_train is True else None)

        # Set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output


class Generator(object):
    def __init__(self, name, dims, norm='batch', _ops=None, logger=None):
        self.name = name
        self.dims = dims
        self.norm = norm
        self._ops = _ops
        self.logger = logger
        self.reuse = False

    def __call__(self, x, is_train=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x)

            # (N, 100) -> (N, 4, 5, 1024)
            h0_linear = tf_utils.linear(x, 4*5*self.dims[0], name='h0_linear', initializer='He',
                                        logger=self.logger if is_train is True else None)
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 5, self.dims[0]])
            h0_norm = tf_utils.norm(h0_reshape, name='h0_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                     logger=self.logger if is_train is True else None)
            h0_relu = tf_utils.relu(h0_norm, name='h0_relu', logger=self.logger if is_train is True else None)

            # (N, 4, 5, 1024) -> (N, 8, 10, 512)
            h1_deconv = tf_utils.deconv2d(h0_relu, output_dim=self.dims[1], name='h1_deconv2d', initializer='He',
                                          logger=self.logger if is_train is True else None)
            h1_norm = tf_utils.norm(h1_deconv, name='h1_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h1_relu = tf_utils.relu(h1_norm, name='h1_relu', logger=self.logger if is_train is True else None)

            # (N, 8, 10, 512) -> (N, 16, 20, 256)
            h2_deconv = tf_utils.deconv2d(h1_relu, output_dim=self.dims[2], name='h2_deconv2d', initializer='He',
                                          logger=self.logger if is_train is True else None)
            h2_norm = tf_utils.norm(h2_deconv, name='h2_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h2_relu = tf_utils.relu(h2_norm, name='h2_relu', logger=self.logger if is_train is True else None)
            # (N, 16, 20, 256) -> (N, 15, 20, 256)
            h2_split, _ = tf.split(h2_relu, [15, 1], axis=1, name='h2_split')
            tf_utils.print_activations(h2_split, logger=self.logger if is_train is True else None)

            # (N, 15, 20, 256) -> (N, 30, 40, 128)
            h3_deconv = tf_utils.deconv2d(h2_split, output_dim=self.dims[3], name='h3_deconv2d', initializer='He',
                                          logger=self.logger if is_train is True else None)
            h3_norm = tf_utils.norm(h3_deconv, name='h3_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h3_relu = tf_utils.relu(h3_norm, name='h3_relu', logger=self.logger if is_train is True else None)

            # (N, 30, 40, 128) -> (N, 60, 80, 64)
            h4_deconv = tf_utils.deconv2d(h3_relu, output_dim=self.dims[4], name='h4_deconv2d', initializer='He',
                                          logger=self.logger if is_train is True else None)
            h4_norm = tf_utils.norm(h4_deconv, name='h4_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train is True else None)
            h4_relu = tf_utils.relu(h4_norm, name='h4_relu', logger=self.logger if is_train is True else None)

            # (N, 60, 80, 64) -> (N, 120, 160, 1)
            h5_deconv = tf_utils.deconv2d(h4_relu, output_dim=self.dims[5], name='h5_deconv', initializer='He',
                                       logger=self.logger if is_train is True else None)
            output = tf_utils.tanh(h5_deconv, name='output', logger=self.logger if is_train is True else None)

            # Set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
