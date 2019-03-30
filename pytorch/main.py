import argparse
from trainer import Trainer
from data import mnist

# arguments
parser = argparse.ArgumentParser()

# data
parser.add_argument("--data_dir", type=str, default='./dataset/mnist', help="size of the batches")
parser.add_argument("--num_threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

# train
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="adam lr")
parser.add_argument("--b1", type=float, default=0.5, help="adam beta 1")
parser.add_argument("--b2", type=float, default=0.999, help="adam beta 2")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")

# model
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--lambda_gp", type=float, default=10.0, help="lambda of gradient penalty")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

# save and monitor
parser.add_argument("--exp_dir", type=str, default='./exp', help="directory for experiments")
parser.add_argument("--exp_name", type=str, default='test', help="name of the experiment")
parser.add_argument("--gen_dir", type=str, default='gen', help="directory for generated images while monitoring")
parser.add_argument("--result_dir", type=str, default='result', help="directory for generated images when finished")
parser.add_argument("--iter_to_print", type=int, default=100, help="number of iterations to print")
parser.add_argument("--iter_to_save", type=int, default=400, help="number of iterations to save model and images")
args = parser.parse_args()

def main():

    # double check
    print(args)

    # data
    dataset = mnist(args.data_dir)

    # init trainer obj
    trainer = Trainer(args)

    # ----- #
    # Train #
    # ----- #
    trainer.train(dataset)

    # --------- #
    # Fine-Tune #
    # --------- #
    # trainer.restore()
    # trainer.train(dataset)

    # -------- #
    # Gen Img  #
    # -------- #
    trainer.gen()


if __name__ == '__main__':
    # run
    main()
