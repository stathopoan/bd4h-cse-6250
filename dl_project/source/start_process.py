import numpy as np
import pandas as pd
from utils import *
import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
from models import BOWPool
import torch.optim as optim
import torch.nn as nn
from my_datasets import WordsWithLabelDataset,visit_collate_fn
from plots import plot_learning_curves, plot_confusion_matrix
from torchsummary import summary

PATH_MY_EMBEDDINGS = "C:\\Users\\TRON\\bd4h\\bd4h-cse-6250\\embeddings.csv"
PATH_VAL_SET = "C:\\Users\\TRON\\bd4h\\bd4h-cse-6250\\val_split.csv"
PATH_TRAIN_SET = "C:\\Users\\TRON\\bd4h\\bd4h-cse-6250\\train_split.csv"
PATH_TEST_SET = "C:\\Users\\TRON\\bd4h\\bd4h-cse-6250\\test_split.csv"
TRAIN_VAL_TEST_PATH = "C:\\Users\\TRON\\bd4h\\bd4h-cse-6250\\train_split.csv"
PATH_OUTPUT = "output"

# Compute dictionaries. Word to index and vice versa.
ind2w, w2ind = get_words_to_indexes_dictionaries(PATH_MY_EMBEDDINGS)
# Compute dictionary with index to code  and vice versa along with code descriptions
ind2c, c2ind, desc_dict = load_full_codes(TRAIN_VAL_TEST_PATH)



NUM_EPOCHS = 10
BATCH_SIZE = 32
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0
save_file = 'MyBaseline.pth'

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device == "cpu":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


df_train = pd.read_csv(PATH_TRAIN_SET)
df_val = pd.read_csv(PATH_VAL_SET)
df_test = pd.read_csv(PATH_TEST_SET)

train_dataset = WordsWithLabelDataset(df_train, c2ind, w2ind)
valid_dataset = WordsWithLabelDataset(df_val, c2ind, w2ind)
test_dataset = WordsWithLabelDataset(df_test, c2ind, w2ind)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)

model = BOWPool(len(ind2c), PATH_MY_EMBEDDINGS)


# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, save_file))

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)