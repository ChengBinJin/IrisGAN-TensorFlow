# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import cv2
import numpy as np


class CASIA_Iris(object):
    def __init__(self, data_path, info=True):
        self.name = data_path.split('/')[-1]
        self.dataset_path = data_path
        self.file_names = []

        for idx, (root, ids, files) in enumerate(os.walk(self.dataset_path)):
            if idx == 0:
                self.num_persons = len(ids)

            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.jpg':
                    file_name = os.path.join(root, filename)
                    self.file_names.append(file_name)

        self.num_imgs = len(self.file_names)
        self.img_shape = cv2.imread(self.file_names[0], cv2.IMREAD_GRAYSCALE).shape

        if info:
            self.print_info()

    def random_batch(self, num_imgs=16, height=480, width=640):
        indexes = np.random.random_integers(low=0, high=self.num_imgs, size=num_imgs)
        img_paths = [self.file_names[idx] for idx in indexes]
        imgs = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in img_paths]

        if imgs[0].shape != (height, width):
            for idx, img in enumerate(imgs):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                imgs[idx] = img

        return imgs

    def print_info(self):
        print('Dataset: {}'.format(self.name))
        print('Num of persons: {}'.format(self.num_persons))
        print('Num of images: {}'.format(self.num_imgs))
        print('Img shape: {}\n'.format(self.img_shape))


if __name__ == '__main__':
    names = ['CASIA-Iris-Distance', 'CASIA-Iris-Interval', 'CASIA-Iris-Lamp',
             'CASIA-Iris-Syn', 'CASIA-Iris-Thousand', 'CASIA-Iris-Twins']
    path = '../../Data/CASIA-IRisV4/'

    save_fold = os.path.join('log', 'imgs')
    if not os.path.isdir(save_fold):
        os.makedirs(save_fold)


    for idx in range(len(names)):
        dataset = CASIA_Iris(data_path=os.path.join(path, names[idx]), info=True)
        imgs = dataset.random_batch(num_imgs=16)

        height, width = imgs[0].shape
        margin = 5
        canvas = np.zeros((4*height+5*margin, 4*width+5*margin), dtype=np.uint8)

        for i, img in enumerate(imgs):
            n_row, n_col = i // 4, i % 4

            # Calculate start and end positions
            h_start = n_row * height + (n_row + 1) * margin
            h_end = (n_row + 1) * height + (n_row + 1) * margin
            w_start = n_col * width + (n_col + 1) * margin
            w_end = (n_col + 1) * width + (n_col + 1) * margin
            canvas[h_start:h_end, w_start:w_end] = imgs[i]

            # Save images
            cv2.imwrite(os.path.join(save_fold, names[idx] + '.png'), canvas)
