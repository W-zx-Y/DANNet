import argparse

# validation set path
DATA_DIRECTORY = '/path/to/Dark_Zurich_val_anon/rgb_anon/val'
DATA_LIST_PATH = './dataset/lists/zurich_val.txt'

# test set path
# DATA_DIRECTORY = '/path/to/public_data_2/rgb_anon'
# DATA_LIST_PATH = './dataset/lists/zurich_test.txt'

IGNORE_LABEL = 255
NUM_CLASSES = 19
SET = 'val'

MODEL = 'PSPNet'
RESTORE_FROM = './trained_models/dannet_psp.pth'
RESTORE_FROM_LIGHT = './trained_models/dannet_psp_light.pth'
SAVE_PATH = './result/dannet_'+MODEL
STD = 0.16


def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-light", type=str, default=RESTORE_FROM_LIGHT,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--std", type=float, default=STD)
    return parser.parse_args()