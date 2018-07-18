import numpy as np

x_phr = np.load('x_phr.npy')[:,:,:,24:108]
x_phr = np.transpose(x_phr, (0, 2, 3, 1)).reshape(-1, 4, 96, 84, 2)
print(x_phr.shape)
np.save('x_phr.npy', x_phr)

x_bar = x_phr[:, 0, :, :, :]
print(x_bar.shape)
np.save('x_bar.npy', x_bar)
