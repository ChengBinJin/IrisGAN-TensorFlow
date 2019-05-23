import os
import sys
import cv2

class Dataset(object):
    def __init__(self, name=None, info=True):
        self.name = name
        self.dataset_path = '../../Data/CASIA-IRisV4/{}'.format(self.name)
        self.file_names = []

        for root, _, files in os.walk(self.dataset_path):
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
        print('Num of images: {}'.format(self.num_imgs))
        print('Img shape: {}'.format(self.img_shape))


if __name__ == '__main__':
    names = ['CASIA-Iris-Interval', 'CASIA-Iris-Twins', 'CASIA-Iris-Lamp']
    dataset = Dataset(name=names[2], info=True)

    for img_name in dataset.file_names:
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(dataset.name, img)
        print('Name: {}'.format(img_name))

        if cv2.waitKey(0) & 0xFF == 27:
            sys.exit('Esc clicked!')

