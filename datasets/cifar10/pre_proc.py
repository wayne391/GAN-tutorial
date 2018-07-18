import pickle
import numpy as np
import glob
import os
from sklearn.utils import shuffle

normalize = lambda data: data/255
denormalize = lambda data: data*255

def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def label_to_onehot(labels, num_class=10):
    labels = np.array(labels)
    onehots = np.zeros((len(labels), num_class), dtype=bool)
    for idx, lab in enumerate(labels):
        onehots[idx, lab] = True
    return onehots

def proc(path_dataset, is_shuflle=False, is_norm=True):

    file_list = [f for f in os.listdir(path_dataset) if 'data_batch' in f]

    image_list = []
    label_list = []

    print('[*] Loading Cifar10...')
    for f in file_list:
        tmp = unpickle(os.path.join(path_dataset,  f))
        image_list.append(tmp['data'])
        label_list.extend(tmp['labels'])

    data = np.concatenate(image_list, axis=0)
    label = label_to_onehot(np.array(label_list))

    if is_shuflle:
        print('[*] Shuffling...')
        data, label = shuffle(data, label, random_state=0)

    if is_norm:
        print('[*] Normalizing...')
        data = normalize(data)

    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)
    print(data.shape)
    print(label.shape)

    return data, label

if __name__ == '__main__':
    root= 'src'
    data, label = proc(root)

    print('[*] saving...')
    np.save('x.npy', data)
    np.save('y.npy', label)