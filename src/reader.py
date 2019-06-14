# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import tensorflow as tf

class Reader(object):
    def __init__(self, tfrecords_file, image_shape=(120, 160, 1), batch_size=1, is_train=True, min_queue_examples=100,
                 num_threads=8, name='DataReader'):
        self.tfrecords_file = tfrecords_file
        self.image_shape = image_shape

        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.is_train = is_train
        self.name = name

        # For data augmentations
        # self.resize_factor = 1.05
        # self.rotatat_angle = 5.

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example, features={
                'image/file_name': tf.FixedLenFeature([], tf.string),
                'image/encoded_image': tf.FixedLenFeature([], tf.string)})

            image_buffer = features['image/encoded_image']
            image_name_buffer = features['image/file_name']
            image = tf.image.decode_jpeg(image_buffer, channels=self.image_shape[2])
            image = self.preprocess(image)

            image_batch, name_batch = tf.train.shuffle_batch([image, image_name_buffer],
                                                             batch_size=self.batch_size,
                                                             num_threads=self.num_threads,
                                                             capacity=self.min_queue_examples + 3 * self.batch_size,
                                                             min_after_dequeue=self.min_queue_examples)

        return image_batch, name_batch

    def preprocess(self, image):
        # Resize to 2D
        img = tf.image.resize_images(image, size=(self.image_shape[0], self.image_shape[1]))

        return img
