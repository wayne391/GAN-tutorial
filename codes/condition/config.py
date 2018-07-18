import os
import numpy as np
import pprint
import importlib



# basic exp info
gpu = '0'
exp_name = 'try/cifar10_sngan_acgan_no'
num_epoch = 60
batch_size = 64


# set models and presets
path_model = 'models.sngan_acgan'
path_preset = 'presets.cifar10'

# runtime import
ModelConfig = importlib.import_module(path_model).ModelConfig
DataConfig = importlib.import_module(path_preset).DataConfig

# original config
config = {
    'path_model': path_model,
    'path_preset': path_preset,
    'gpu': gpu,
    'exp_name': exp_name,
    'num_epoch': num_epoch,
    'verbose': True,
    'sample_along_training': lambda cnt: (cnt%500 == 0) or
                                (cnt < 500 and cnt%50 == 0),

    'optimizer': {
        # Parameters for the Adam optimizers
        'lr': 0.0002,
        'beta1': 0,
        'beta2': .9,
        'epsilon': 1e-8
    },

    # Samples
    'num_sample': 64,
    'sample_grid': (8, 8),

    # Dirs
    'dirs': {
        'ckpt': 'checkpoints',
        'sample': 'samples',
        'log': 'logs',
        'src': 'src',
        # 'eval': 'eval',
        # 'survey': 'survey',
    },

    # Parameters
    'batch_size': batch_size,
}

# re-arrange path of folders
for key, path in config['dirs'].items():
    path = os.path.join(config['exp_name'], path)
    config['dirs'][key] = path


# load specific configs
def class_to_dict(x):
    return dict((key, value) for key, value in x.__dict__.items()
                if not key.startswith('__'))

config_m = class_to_dict(ModelConfig)
config_d = class_to_dict(DataConfig)

# print('*model config:', config_m)
# print('*data config:', config_d)

for k, v in config_m.items():
   config[k] = v

for k, v in config_d.items():
   config[k] = v

# Done!!
print('*final config:')
# pprint.pprint(config)
