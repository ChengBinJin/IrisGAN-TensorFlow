# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import numpy as np
import tensorflow as tf

from CASIA_iris import CASIA_Iris


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input_data', '../../Data/CASIA-IRisV4/CASIA-Iris-Thousand',
                       'data input directory, default: ../../Data/CASIA-IrisV4/CASIA-Iris-Thousand')
tf.flags.DEFINE_string('output_data', '../../Data/CASIA-IrisV4/CASIA-Iris-Thousand',
                       'data output directory, default: ../../Data/CASIA-IrisV4/CASIA-Iris-Thousand')


def data_writer(input_dir, output_name):
    dataset = CASIA_Iris(data_path=input_dir)
    file_paths = dataset.file_names
    num_imgs = len(file_paths)

    # Create tfrecrods dir if not exists
    output_file = '{}.tfrecords'.format(output_name)
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for idx in range(num_imgs):
        img_path = file_paths[idx]

        with tf.gfile.FastGFile(img_path, 'rb') as f:
            img_data = f.read()

        example = _convert_to_example(img_path, img_data)
        writer.write(example.SerializeToString())

        if np.mod(idx, 100) == 0:
            print('Processed {}/{}...'.format(idx, num_imgs))

    print('Finished!')
    writer.close()


def _convert_to_example(img_path, img_buffer):
    # Build an example proto
    img_name = os.path.basename(img_path)

    example = tf.train.Example(features=tf.train.Features(
        feature={'image/file_name': _bytes_feature(tf.compat.as_bytes(img_name)),
                 'image/encoded_image': _bytes_feature(img_buffer)}))

    return example

def _bytes_feature(value):
    # Wrapper for inserting bytes features into example proto
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    print("Convert {} data to tfrecrods...".format(FLAGS.input_data))
    data_writer(FLAGS.input_data, FLAGS.output_data)


if __name__ == '__main__':
    tf.app.run()
