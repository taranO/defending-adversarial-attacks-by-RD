"""Defending against adversarial attacks by randomized diversification"""

import argparse
import keras
import os

from setup_mnist import *
from setup_cifar import *
import libs.model_multi_channel as mcm

########################################################################################################################

def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

########################################################################################################################
parser = argparse.ArgumentParser(description="Train multi-channel system with randomized diversification.")
parser.add_argument("--type", default="mnist", help="The dataset.")
parser.add_argument("--save_to", default="models", help="Path where to save models.")
parser.add_argument("--is_zero", default=False, type=int, help="Is to use hard thresholding.")
parser.add_argument("--epochs", default=50, type=int, help="The number of epochs.")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--oprimazer", default="adam", help="The oprimazation technique.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")

args = parser.parse_args()

########################################################################################################################
if __name__ == "__main__":

    P = [1, 2, 3]  # number of channels per subband
    SUBBANDS = ["d", "h", "v"]  # DCT subbands
    SAVE_TO_DIR = makeDir(args.save_to + "/" + args.type)

    with tf.Session() as sess:
        keras.backend.set_session(sess)

        # training data and parameters
        if args.type == "mnist":
            data = MNIST()
            nn_param = [32, 32, 64, 64, 200, 200]
            model = MNISTModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 28
            N_CHANELS = 1
        elif args.type == "fashion_mnist":
            data = FashionMNIST()
            nn_param = [32, 32, 64, 64, 200, 200]
            model = MNISTModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 28
            N_CHANELS = 1
        elif args.type == "cifar":
            data = CIFAR()
            nn_param = [64, 64, 128, 128, 256, 256]
            model = CIFARModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 32
            N_CHANELS = 3

        # multi-channel model initialization with classifier defined in model variable
        multi_channel_model = mcm.MultiChannel(model,
                                               type          = args.type,
                                               epochs        = args.epochs,
                                               optimazer     = args.oprimazer,
                                               learning_rate = args.lr,
                                               batch_size    = args.batch_size,
                                               permt         = P,
                                               subbands      = SUBBANDS,
                                               model_dir     = SAVE_TO_DIR,
                                               img_size      = IMAGE_SIZE,
                                               img_channels  = N_CHANELS)
        # multi-channel model training
        multi_channel_model.train(data)






















