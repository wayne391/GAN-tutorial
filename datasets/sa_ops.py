import SharedArray as sa
import sys
import numpy as np
import os
import time
import datetime
import argparse

def create(sa_name, npy_name):
    print(sa_name, npy_name)
    print('[*] Loading...')
    X = np.load(npy_name)
    print(X.shape)

    print('[*] Saving...')
    tmp_arr = sa.create(sa_name, X.shape)
    np.copyto(tmp_arr, X)

def listing():
    for n in sa.list():
        print(n)

    print('%d shared arrays'%len(sa.list()))

def delete(sa_name):
    sa.delete(sa_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="python sa_ops.py <sa_name> <npy_name>")

    parser.add_argument("-c", "--create", nargs='*', default=True)
    parser.add_argument("-l", "--list", action="store_true", default=False)
    parser.add_argument("-d", "--delete", dest="delete", default=False)
    args = vars(parser.parse_args())


    if isinstance(args['create'], list):
        if len(args['create']) is 2:
                create(args['create'][0], args['create'][1])
        else:
            raise IOError('2 args expected, <sa_name> <npy_name>')

    elif args['delete']:
        delete(args['delete'])
    elif args['list']:
        listing()
    else:
        raise ValueError('python sa_ops.py <sa_name> <npy_name>')
