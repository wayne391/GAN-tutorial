denormalize = lambda data: data*255
path_dataset = 'wayne_cifar10_x'

class ModelConfig:
    'height': 32,
    'width': 32,
    'channel': 3,
    'path_dataset': path_dataset,
    'denorm': denormalize,