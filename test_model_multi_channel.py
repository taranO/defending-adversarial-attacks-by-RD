"""Defending against adversarial attacks by randomized diversification"""

import argparse
import keras

from setup_mnist import *
from setup_cifar import *
import libs.model_multi_channel as mcm

########################################################################################################################
parser = argparse.ArgumentParser(description="Train multi-channel system with randomized diversification.")
parser.add_argument("--type", default="mnist", help="The dataset type.")
parser.add_argument("--data_dir", default="data/attacked", help="The dataset path.")
parser.add_argument("--model_dir", default="models", help="Path where to save models.")
parser.add_argument("--is_zero", default=False, type=int, help="Is to use hard thresholding.")
parser.add_argument("--epoch", default=10, type=int, help="The number of epochs.")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--samples", default=1000, type=int, help="The number of test samples.")
parser.add_argument("--attack_type", default="carlini_l2", help="The attack type.")


args = parser.parse_args()

########################################################################################################################

def prepare_data_for_classification(data, IMAGE_SIZE, N_CHANELS):
    test = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANELS)

    if np.max(test) > 0.75:
        test -= 0.5

    return test


if __name__ == "__main__":

    P = [1,2,3]  # number of channels per subband
    SUBBANDS = ["d", "h", "v"]  # DCT subbands
    MODEL_DIR = args.model_dir + "/" + args.type
    EPOCHS = [args.epoch for i in range(len(P)*len(SUBBANDS))]

    pref = "adv"
    DATA_DIR = args.data_dir + "/" + args.type + "/" + args.attack_type

    with tf.Session() as sess:
        keras.backend.set_session(sess)

        # test data and parameters
        if args.type == "mnist":
            nn_param = [32, 32, 64, 64, 200, 200]
            model = MNISTModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 28
            N_CHANELS = 1
        elif args.type == "fashion_mnist":
            nn_param = [32, 32, 64, 64, 200, 200]
            model = MNISTModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 28
            N_CHANELS = 1
        elif args.type == "cifar":
            nn_param = [64, 64, 128, 128, 256, 256]
            model = CIFARModelAllLayers(nn_param, session=sess)
            IMAGE_SIZE = 32
            N_CHANELS = 3

        # multi-channel model initialization with classifier defined in model variable
        multi_channel_model = mcm.MultiChannel(model,
                                               type          = args.type,
                                               permt         = P,
                                               subbands      = SUBBANDS,
                                               model_dir     = MODEL_DIR,
                                               img_size      = IMAGE_SIZE,
                                               img_channels  = N_CHANELS)
        multi_channel_model.test_init(sess, nn_param, EPOCHS)


        # --------------------------------------------------------------------------------------------------------------
        N = 0
        org_labels  = [] # original labels
        pred_labels = [] # labels predicted by multi-channel model
        for i in range(args.samples):

            label = np.load(DATA_DIR + "/labels_%d.npy" % i)
            input = np.load(DATA_DIR + "/%s_img_%d.npy" % (pref, i))
            input = prepare_data_for_classification(input, IMAGE_SIZE, N_CHANELS)
            N += input.shape[0]

            prediction = multi_channel_model.predict(input)

            if i == 0:
                pred_labels = prediction
            else:
                pred_labels = np.vstack((pred_labels, prediction))
            org_labels.append(label)

        # classificaiton error
        diff = org_labels - np.reshape(pred_labels.argmax(1), (args.samples, -1))
        diff[diff != 0] = 1
        total_error = 100 * np.sum(diff) / N

        print('datatype = %s, attack_type = %s: \t error = %0.2f' %  (args.type, args.attack_type, total_error))
