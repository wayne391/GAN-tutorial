import os
import sys
import moviepy.editor as mpy
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
from skimage.io import imsave


def read_mnist_data(data_dir, is_shuffle=False):
    mnist_path = os.path.join(data_dir, 'mnist')
    mnist = input_data.read_data_sets(mnist_path, one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    
    if is_shuffle:
        x_train, y_train = shuffle(x_train, y_train)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    
    return x_train, y_train, x_test, y_test


def data_generator(x, y=None, batch_size=32, is_label=True):
    batches = np.ceil(x.shape[0] / batch_size).astype(int)
    while True:
        for i in range(batches):
            yield i, x[i * batch_size : (i+1) * batch_size],\
                     y[i * batch_size : (i+1) * batch_size] if y is not None else None


def show_result(batch_res, fname, img_size=(28, 28), grid_size=(8, 8), grid_pad=5):
    img_width, img_height = img_size
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
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
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)
    return img_grid

    
def make_gif(images, fname, duration=2, true_image=False):
    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=(len(images) / duration))


class ProgressBar(object):
    def __init__(self, iteration, total, prefix='', suffix='', decimals=3,
                 length=10, fillchr='=', is_percent=False):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (int)
            total       - Required  : total iterations (int)
            prefix      - Optional  : prefix string (str)
            suffix      - Optional  : suffix string (str)
            decimals    - Optional  : positive number of decimals in percent complete (int)
            length      - Optional  : character length of bar (int)
            fill        - Optional  : bar fill character (str) # '█'
        """
        self._iteration = iteration
        self._total = total
        self._prefix = prefix
        self._suffix = suffix
        self._decimals = decimals
        self._length = length
        self._fillchr = fillchr
        self._is_percent = is_percent
    
    def update(self, iteration, prefix=None, suffix=None):
        prefix = self._prefix if prefix is None else prefix
        suffix = self._suffix if suffix is None else suffix
        filledlen = int(self._length * iteration // self._total)
        bar = self._fillchr * filledlen + '-' * (self._length - filledlen)
        if self._is_percent:
            postfix = ("{0:." + str(self._decimals) + "f}").format(100 * (iteration / float(self._total)))
            print('\r%s [%s] %s%% %s' % (prefix, bar, postfix, suffix), end = '\r')
        else:
            postfix = ("[%5d/%d]" % (iteration, self._total))
            print('\r%s [%s] %s %s' % (prefix, bar, postfix, suffix), end = '\r')
        if iteration == self._total: 
            print()