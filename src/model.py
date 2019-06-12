import tensorflow as tf

import tensorflow_utils as tf_utils
import utils as utils

class DCGAN(object):
    def __init__(self, data_shape, lr=2e-4, total_iters=2e5, is_train=True, log_dir=None, name='dcgan'):
        self.data_shape = data_shape
        self.gen_dims = [1024, 512, 256, 128, 64, 1]
        self.dis_dims = [64, 128, 256, 512, 1024, 1]
        self.lr = lr
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.is_train = is_train
        self.log_dir = log_dir
        self.name = name
        self.logger, self.file_handler, self.stream_handler = utils.init_logger(log_dir=self.log_dir,
                                                                                is_train=self.is_train,
                                                                                name=self.name)
        self.gen_ops, self.dis_ops = [], []

        with tf.variable_scope(self.name):
            self._build_net()

        # self._tensorboard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _build_net(self):
        self.mode_tfph = tf.placeholder(dtype=tf.bool, name='mode_tfph')

        self.gen = Generator(name='gen', dims=self.gen_dims, norm='batch', _ops=self.gen_ops, logger=self.logger)
        self.dis = Discriminator(name='dis', dims=self.dis_dims, norm='batch', _ops=self.dis_ops, logger=self.logger)

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
            h0_conv = tf_utils.conv2d(x, output_dim=self.dims[0], initializer='he',
                                      logger=self.logger if is_train else None)
            h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu', logger=self.logger if is_train else None)

            # (N, 60, 80, 64) -> (N, 30, 40, 128)
            h1_conv = tf_utils.conv2d(h0_lrelu, output_dim=self.dims[1], initializer='he',
                                      logger=self.logger if is_train else None)
            h1_norm = tf_utils.norm(h1_conv, name='h1_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h1_lrelu = tf_utils.lrelu(h1_norm, name='h1_lrelu', logger=self.logger if is_train else None)

            # (N, 30, 40, 128) -> (N, 15, 20, 256)
            h2_conv = tf_utils.conv2d(h1_lrelu, output_dim=self.dims[2], initializer='he',
                                      logger=self.logger if is_train else None)
            h2_norm = tf_utils.norm(h2_conv, name='h2_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h2_lrelu = tf_utils.lrelu(h2_norm, name='h2_lrelu', logger=self.logger if is_train else None)

            # (N, 15, 20, 256) -> (N, 8, 10, 512)
            h3_conv = tf_utils.conv2d(h2_lrelu, output_dim=self.dims[3], initializer='he',
                                      logger=self.logger if is_train else None)
            h3_norm = tf_utils.norm(h3_conv, name='h3_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h3_lrelu = tf_utils.lrelu(h3_norm, name='h3_lrelu', logger=self.logger if is_train else None)

            # (N, 8, 10, 512) -> (N, 4, 5, 1024)
            h4_conv = tf_utils.conv2d(h3_lrelu, output_dim=self.dims[4], initializer='he',
                                      logger=self.logger if is_train else None)
            h4_norm = tf_utils.norm(h4_conv, name='h4_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h4_lrelu = tf_utils.lrelu(h4_norm, name='h4_lrelu', logger=self.logger if is_train else None)
            # (N, 4, 5, 1024) -> (N, 4*5*1024)
            h4_flatten = tf_utils.flatten(h4_lrelu, name='h4_flatten', logger=self.logger if is_train else None)

            # (N, 4*5*1024) -> (N, 1)
            output = tf_utils.linear(h4_flatten, output_size=self.dims[5], initializer='he', name='output',
                                     logger=self.logger if is_train else None)

            # Set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name=self.name)

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
                                        logger=self.logger if is_train else None)
            h0_reshape = tf.reshape(h0_linear, [tf.shape(h0_linear)[0], 4, 5, self.dims[0]])
            h0_norm = tf_utils.norm(h0_reshape, name='h0_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                     logger=self.logger if is_train else None)
            h0_relu = tf_utils.relu(h0_norm, name='h0_relu', logger=self.logger if is_train else None)

            # (N, 4, 5, 1024) -> (N, 8, 10, 512)
            h1_deconv = tf_utils.deconv2d(h0_relu, output_dim=self.dims[1], name='h1_deconv2d', initializer='He',
                                          logger=self.logger if is_train else None)
            h1_norm = tf_utils.norm(h1_deconv, name='h1_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h1_relu = tf_utils.relu(h1_norm, name='h1_relu', logger=self.logger if is_train else None)

            # (N, 8, 10, 512) -> (N, 16, 20, 256)
            h2_deconv = tf_utils.deconv2d(h1_relu, output_dim=self.dims[2], name='h2_deconv2d', initializer='He',
                                          logger=self.logger if is_train else None)
            h2_norm = tf_utils.norm(h2_deconv, name='h2_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h2_relu = tf_utils.relu(h2_norm, name='h2_relu', logger=self.logger if is_train else None)
            # (N, 16, 20, 256) -> (N, 15, 20, 256)
            h2_split, _ = tf.split(h2_relu, [15, 1], axis=1, name='h2_split')
            tf_utils.print_activations(h2_split, logger=self.logger if is_train else None)

            # (N, 15, 20, 256) -> (N, 30, 40, 128)
            h3_deconv = tf_utils.deconv2d(h2_split, output_dim=self.dims[3], name='h3_deconv2d', initializer='He',
                                          logger=self.logger if is_train else None)
            h3_norm = tf_utils.norm(h3_deconv, name='h3_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h3_relu = tf_utils.relu(h3_norm, name='h3_relu', logger=self.logger if is_train else None)

            # (N, 30, 40, 128) -> (N, 60, 80, 64)
            h4_deconv = tf_utils.deconv2d(h3_relu, output_dim=self.dims[4], name='h4_deconv2d', initializer='He',
                                          logger=self.logger if is_train else None)
            h4_norm = tf_utils.norm(h4_deconv, name='h4_batch', _type='batch', _ops=self._ops, is_train=is_train,
                                    logger=self.logger if is_train else None)
            h4_relu = tf_utils.relu(h4_norm, name='h4_relu', logger=self.logger if is_train else None)

            # (N, 60, 80, 64) -> (N, 120, 160, 1)
            output = tf_utils.deconv2d(h4_relu, output_dim=self.dims[5], name='output', initializer='He',
                                       logger=self.logger if is_train else None)

            # Set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
