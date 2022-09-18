import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
from PIL import Image
from six.moves.urllib.request import urlretrieve
from six.moves import xrange  # pylint: disable=redefined-builtin
from sklearn.utils import shuffle
import _pickle as pickle

image_size = 64  # Pixel width and height.
num_channels = 3
num_classes = 4


def load(data_path, max_num_images):
    dataset = np.ndarray(
        shape=(max_num_images, image_size, image_size, num_channels, 2), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images, num_classes), dtype=np.int32)

    with open(data_path + '/diag_dict.pickle', 'rb') as handle:
        diag_dict = pickle.load(handle)

    label_index = 0
    image_index = 0
    for key in diag_dict.keys():
        imageA_filenames = data_path + '/' + str(key) + '_1.jpg'
        imageB_filenames = data_path + '/' + str(key) + '_2.jpg'
        print(key)
        try:
            imageA = Image.open(imageA_filenames)
            imageB = Image.open(imageB_filenames)
            imageA = imageA.resize((image_size, image_size))
            imageB = imageB.resize((image_size, image_size))
            imageA = np.asarray(imageA)
            imageB = np.asarray(imageB)
            imageA_mean = 0
            imageB_mean = 0
            imageA_std = 256
            imageB_std = 256
            imageA = (imageA - imageA_mean) / imageA_std
            imageB = (imageB - imageB_mean) / imageB_std
            dataset[image_index, :, :, :, 0] = imageA
            dataset[image_index, :, :, :, 1] = imageB
            image_index += 1
            labels[label_index] = diag_dict[key]

            label_index += 1

        except IOError as e:
            print('Could not read:', imageA_filenames, ':', e, '- it\'s ok, skipping.')
        if image_index >= max_num_images - 1:
            print('reached maximum number of images allowed!')
            break

    assert image_index == label_index, (
            'images.shape: %s labels.shape: %s ' % (image_index,
                                                    label_index))
    num_images = image_index
    num_labels = label_index
    if num_labels != num_images:
        raise Exception('num of images %d is not squal to the number of label files', num_images, num_labels)
    dataset = dataset[0:num_images]
    labels = labels[0:num_labels]
    dataset, labels = shuffle(dataset, labels)
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    print('Labels:', labels.shape)
    return dataset, labels
