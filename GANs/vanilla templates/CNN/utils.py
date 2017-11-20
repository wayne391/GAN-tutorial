from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
from skimage.io import imsave
import numpy as np

def read_input(dataset_dir, is_shuffle=False):
    mnist = input_data.read_data_sets(dataset_dir, one_hot = True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    x_train = x_train.reshape((-1,28,28,1))
    x_test = x_test.reshape((-1,28,28,1))
    x_train = x_train.astype(np.float32)
    if is_shuffle:
        x_train, y_train = shuffle(x_train, y_train)
    
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    return x_train, y_train, x_test, y_test

def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = batch_res.reshape((batch_res.shape[0], 28, 28))
    print(np.min(batch_res))
    print(np.max(batch_res))
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img.reshape(28, 28)
    imsave(fname, img_grid)

