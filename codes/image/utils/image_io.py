import numpy as np
import os
from scipy.misc import imsave


def recolor(x, colormap, post_proc=None):
    '''
    recolor the input matrix with arbitrary number of channel
    '''
    image_shape = x.shape
    x_re = np.matmul(x.reshape(-1, image_shape[-1]), colormap). \
                    reshape(list(image_shape[:3])+[3])
    return post_proc(x_re)

def save_result(batch_res, grid_size, path='./', name='sample',
        grid_pad=5, denorm_func=None, colormap=None, thres=None, transpose=True):

    shape = batch_res.shape # num_sample, img_h, img_w, img_c
    num_sample = shape[0]
    image_shape = list(shape[1:])

    img_h = shape[1]
    img_w = shape[2]
    img_c = shape[3]


    print(np.max(batch_res), np.min(batch_res))
    if thres is not None:
        batch_res = batch_res>=thres

    if colormap is not None:
        batch_res = recolor(batch_res, colormap, post_proc=lambda x: 1.0-x)
        img_c = 3
        image_shape[-1] = 3

    if  denorm_func:
        batch_res = denorm_func(batch_res)


    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)

    img_grid = np.zeros((grid_h, grid_w, img_c), dtype=np.uint8)

    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = res.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w, :] = \
                img.reshape(image_shape)

    if  img_c == 1:
        img_grid = np.squeeze(img_grid)
    imsave(os.path.join(path, name+'.png'), img_grid)

