from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def proc(path_dataset, one_hot=True):

    mnist = input_data.read_data_sets(path_dataset, one_hot=one_hot)

    train_x = mnist.train.images
    train_y = mnist.train.labels

    train_x = train_x.reshape(len(train_x), 28, 28, 1)

    return train_x, train_y



if __name__ == '__main__':
    root = 'src'
    x, y = proc(root)

    print('shape of x:', x.shape, 'range:', np.min(x), '~', np.max(x))
    print('shape of y:', y.shape)

    np.save('x.npy', x)
    np.save('y.npy', y)
