# !/usr/bin/env python2

from __future__ import print_function

import os
from random import randint

import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split

data_path = 'dataset/'

image_rows = 80
image_cols = 80
test_percentage = 0.10


def create_train_and_test_data():
    train_data_path = os.path.join(data_path, 'hands1000')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs_8bit = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    ids = []
    imgs_gt = []

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    dictGT = {}
    with open("dataset/hands1000.csv", 'rb') as features:
        train = features.readlines()
        for i, line in enumerate(train):
            if (i != 0):
                f_info = line.decode().split(',')
                dictGT[f_info[-2].split('/')[-1].split('.')[0]] = f_info[-1]
    features.close()

    for idx, image_name in enumerate(images):
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        imgs_8bit[idx] = np.array([img])
        ids.append(image_name.split('.')[0])
        imgs_gt.append(dictGT[image_name.split('.')[0]])

        if idx % 100 == 0:
            print('Done: {0}/{1} images'.format(idx, total))
    print('Loading done.')

    ids_array = np.array(ids, dtype=object)
    imgs_gt_array = np.array(imgs_gt, dtype=object)
    image_position = np.arange(total)

    image_position_train, image_position_test = \
        train_test_split(image_position, test_size=test_percentage, random_state=randint(0, 100))

    ids_train = create_subarray(image_position_train, ids_array)
    ids_test = create_subarray(image_position_test, ids_array)

    imgs_train_8bit = create_subarray(image_position_train, imgs_8bit)
    imgs_test_8bit = create_subarray(image_position_test, imgs_8bit)

    imgs_gt_train = create_subarray(image_position_train, imgs_gt_array)
    imgs_gt_test = create_subarray(image_position_test, imgs_gt_array)

    np.save('imgs_train_8bit.npy', imgs_train_8bit)
    np.save('imgs_test_8bit.npy', imgs_test_8bit)
    np.save('imgs_train_gt.npy', imgs_gt_train)
    np.save('imgs_test_gt.npy', imgs_gt_test)
    np.save('ids_train.npy', ids_train)
    np.save('ids_test.npy', ids_test)
    print('Saving to .npy files done.')


def create_subarray(subset_index, total_array):
    result_array = []
    for value in subset_index:
        result_array.append(total_array[value])
    return result_array


def load_train_data():
    imgs_train = np.load('imgs_train_8bit.npy')
    imgs_train_gt = np.load('imgs_train_gt.npy')
    imgs_train_id = np.load('ids_train.npy')
    return imgs_train, imgs_train_gt, imgs_train_id


def load_test_data(bit_image):
    imgs_test = np.load('imgs_test_8bit.npy')
    imgs_test_gt = np.load('imgs_test_gt.npy')
    imgs_test_id = np.load('ids_test.npy')

    return imgs_test, imgs_test_gt, imgs_test_id


if __name__ == '__main__':
    create_train_and_test_data()
