import argparse

MODEL = 'PSPNet'  # PSPNet, DeepLab, RefineNet
RESTORE_FROM = './pretrained_models/pretrain_pspnet_150000.pth'

BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 2

SET = 'train'
DATA_DIRECTORY = '/path/to/cityscape'
DATA_LIST_PATH = './dataset/lists/cityscapes_train.txt'
INPUT_SIZE = '512'
DATA_DIRECTORY_TARGET = '/path/to/Dark_Zurich_train_anon/rgb_anon'
DATA_LIST_PATH_TARGET = './dataset/lists/zurich_dn_pair_train.csv'
INPUT_SIZE_TARGET = '960'

NUM_CLASSES = 19
IGNORE_LABEL = 255

LEARNING_RATE = 2.5e-4
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4

NUM_STEPS = 50000
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'+MODEL
STD = 0.05


def get_arguments():
    parser = argparse.ArgumentParser(description="DANNet")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=int, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with pimgolynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--std", type=float, default=STD)
    return parser.parse_args()
