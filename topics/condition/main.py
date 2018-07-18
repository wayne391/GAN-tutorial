import SharedArray as sa
from models.sngan import *
import importlib
import os
from config import config
from trainer import *


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True

    with tf.Session(config=config_tf) as sess:
        x_train = sa.attach(config['path_x'])
        y_train = sa.attach(config['path_y'])


        model_file = importlib.import_module(config['path_model'])
        model = model_file.Net(config)

        trainer = Trainer(sess, model, config)
        trainer.load()
        trainer.train(x_train, y_train)
        #
        # trainer.gen()

if __name__ == '__main__':
    main()