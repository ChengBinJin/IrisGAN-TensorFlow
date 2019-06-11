# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os

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

