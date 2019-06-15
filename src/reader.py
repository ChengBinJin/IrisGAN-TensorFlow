# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import math
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
        self.resize_factor = 1.1
        self.rotate_angle = 5.

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

            img_ori, img_trans, img_flip, img_rotate = self.preprocess(image, is_train=self.is_train)

        return tf.train.shuffle_batch(tensors=[img_ori, img_trans, img_flip, img_rotate, image_name_buffer],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + 3 * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)


    def preprocess(self, image, is_train=True):
        # Resize to 2D
        img_ori = tf.image.resize_images(image, size=(self.image_shape[0], self.image_shape[1]))

        # Data augmentation
        if is_train:
            img_trans = self.RandomTranslation(img_ori)     # Random translation
            img_flip = self.RandomFlip(img_trans)           # Random left-right flip
            img_rotate = self.RandomRotation(img_flip)      # Random rotation

        else:
            img_trans = img_flip = img_rotate = img_ori

        return img_ori, img_trans, img_flip, img_rotate

    def RandomTranslation(self, img_ori):
        # Step 1: Resized to the bigger image
        img = tf.image.resize_images(images=img_ori,
                                     size=(int(self.resize_factor * self.image_shape[0]),
                                           int(self.resize_factor * self.image_shape[1])),
                                     method=tf.image.ResizeMethod.BICUBIC)
        # Step 2: Random crop
        img = tf.image.random_crop(value=img, size=self.image_shape)

        # Step3: Clip value in the range of v_min and v_max
        img = tf.clip_by_value(t=img, clip_value_min=0., clip_value_max=255.)

        return img

    @staticmethod
    def RandomFlip(img_ori, is_random=True):
        if is_random:
            img = tf.image.random_flip_left_right(image=img_ori)
        else:
            img = tf.image.flip_left_right(img_ori)

        return img

    def RandomRotation(self, img_ori):
        radian_min = -self.rotate_angle * math.pi / 180.
        radian_max = self.rotate_angle * math.pi / 180.
        random_angle = tf.random.uniform(shape=[1], minval=radian_min, maxval=radian_max)
        img = tf.contrib.image.rotate(images=img_ori, angles=random_angle, interpolation='BILINEAR')

        return img
