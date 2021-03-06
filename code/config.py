import argparse
import torch

PATH_MY_EMBEDDINGS = "../data/word_embeddings.csv"
PATH_VAL_SET = "../data/top_50/val_split.csv"
PATH_VAL_SET_50 = "../data/top_50/top_50_val_split.csv"
PATH_TRAIN_SET = "../data/top_50/train_split.csv"
PATH_TRAIN_SET_50 = "../data/top_50/top_50_train_split.csv"
PATH_TEST_SET = "../data/top_50/test_split.csv"
PATH_TEST_SET_50 = "../data/top_50/top_50_test_split.csv"
TRAIN_VAL_TEST_PATH = "../data/top_50/train_split.csv"
TRAIN_VAL_TEST_PATH_50 = "../data/top_50/top_50_train_split.csv"
PATH_TOP50_CODES = '../data/top_50/TOP_50_CODES.csv'
PATH_LOADERS = '../data/'
PATH_OUTPUT = '../models/'

USE_TOP50 = True

NUM_EPOCHS = 20
BATCH_SIZE = 32
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 0
patience = 3  # early stopping patience; how long to wait after last time validation loss improved.
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam

# initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dataprep", help="run data prep", action="store_true")
parser.add_argument("-m", "--modeltype", help="choose model type", nargs=1, choices=['lr', 'cnn_attn', 'rnn', 'vanilla_cnn'],
                    default='cnn_attn')
parser.add_argument("-t", "--train", help="choose train flag", action="store_true")
