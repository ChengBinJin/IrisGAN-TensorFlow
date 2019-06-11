# ---------------------------------------------------------
# Tensorflow Iri-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import cv2

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

    def print_info(self):
        print('Dataset: {}'.format(self.name))
        print('Num of persons: {}'.format(self.num_persons))
        print('Num of images: {}'.format(self.num_imgs))
        print('Img shape: {}'.format(self.img_shape))


if __name__ == '__main__':
    names = ['CASIA-Iris-Distance', 'CASIA-Iris-Interval', 'CASIA-Iris-Lamp',
             'CASIA-Iris-Syn', 'CASIA-Iris-Thousand', 'CASIA-Iris-Twins']
    path = '../../Data/CASIA-IRisV4/'
    dataset = CASIA_Iris(data_path=os.path.join(path, names[4]), info=True)

    for img_name in dataset.file_names:
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(dataset.name, img)
        print('Name: {}'.format(img_name))

        if cv2.waitKey(0) & 0xFF == 27:
            sys.exit('Esc clicked!')

