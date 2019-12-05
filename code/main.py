import numpy as np
import pandas as pd
from utils import *
import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
# from models import BOWPool
import torch.optim as optim
import torch.nn as nn
from my_datasets import WordsWithLabelDataset, visit_collate_fn
from models import *
from config import *
import pickle
from plots import *

# Compute dictionaries. Word to index and vice versa.
ind2w, w2ind = get_words_to_indexes_dictionaries(PATH_MY_EMBEDDINGS)

train_val_test_path = None
path_train_set = None
path_val_set = None
path_test_set = None

if USE_TOP50:
    train_val_test_path = TRAIN_VAL_TEST_PATH_50
    path_train_set = PATH_TRAIN_SET_50
    path_val_set = PATH_VAL_SET_50
    path_test_set = PATH_TEST_SET_50
else:
    train_val_test_path = TRAIN_VAL_TEST_PATH
    path_train_set = PATH_TRAIN_SET
    path_val_set = PATH_VAL_SET
    path_test_set = PATH_TEST_SET

# Compute dictionary with index to code  and vice versa along with code descriptions
ind2c, c2ind, desc_dict = load_full_codes(train_val_test_path)

top_50_codes = pd.read_csv(PATH_TOP50_CODES, header=None)
top_50_codes.columns = ['codes_50']

# c2ind_filtered = {}
# count = 0
# for i in top_50_codes['codes_50'].values:
#     c2ind_filtered[i] = count
#     count += 1
#
# ind2c_filtered = {}
# for i, j in c2ind_filtered.items():
#     ind2c_filtered[j] = i

# read arguments from the command line
args = parser.parse_args()
print(args)

# if args.dataprep:
#
#     df_train = pd.read_csv(PATH_TRAIN_SET)
#     df_val = pd.read_csv(PATH_VAL_SET)
#     df_test = pd.read_csv(PATH_TEST_SET)
#
#     if USE_TOP50:
#         for i in range(df_train.shape[0]):
#             df_train.iloc[i, 3] = ";".join(
#                 set(top_50_codes['codes_50']).intersection(set(df_train.iloc[i, 3].split(";"))))
#
#         for i in range(df_val.shape[0]):
#             df_val.iloc[i, 3] = ";".join(set(top_50_codes['codes_50']).intersection(set(df_val.iloc[i, 3].split(";"))))
#
#         for i in range(df_test.shape[0]):
#             df_test.iloc[i, 3] = ";".join(
#                 set(top_50_codes['codes_50']).intersection(set(df_test.iloc[i, 3].split(";"))))
#
#         df_train_filtered = df_train.loc[~(df_train['LABELS'] == ''), :]
#         df_val_filtered = df_val.loc[~(df_val['LABELS'] == ''), :]
#         df_test_filtered = df_test.loc[~(df_test['LABELS'] == ''), :]
#
#         train_dataset = WordsWithLabelDataset(df_train_filtered, c2ind_filtered, w2ind)
#         valid_dataset = WordsWithLabelDataset(df_val_filtered, c2ind_filtered, w2ind)
#         test_dataset = WordsWithLabelDataset(df_test_filtered, c2ind_filtered, w2ind)
#
#     else:
#
#         train_dataset = WordsWithLabelDataset(df_train, c2ind, w2ind)
#         valid_dataset = WordsWithLabelDataset(df_val, c2ind, w2ind)
#         test_dataset = WordsWithLabelDataset(df_test, c2ind, w2ind)
#
#     train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn,
#                               num_workers=NUM_WORKERS)
#     valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn,
#                               num_workers=NUM_WORKERS)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn,
#                              num_workers=NUM_WORKERS)
#
#     with open(PATH_LOADERS + "train_loader.pkl", 'wb') as train_loader_pkl:
#         pickle.dump(train_loader, train_loader_pkl)
#
#     with open(PATH_LOADERS + "test_loader.pkl", 'wb') as test_loader_pkl:
#         pickle.dump(test_loader, test_loader_pkl)
#
#     with open(PATH_LOADERS + "valid_loader.pkl", 'wb') as valid_loader_pkl:
#         pickle.dump(valid_loader, valid_loader_pkl)
#
# else:
#     pass
# with open(PATH_LOADERS + "train_loader.pkl", 'rb') as train_loader_pkl:
#     train_loader = pickle.load(train_loader_pkl)
#
# with open(PATH_LOADERS + "test_loader.pkl", 'rb') as test_loader_pkl:
#     test_loader = pickle.load(test_loader_pkl)
#
# with open(PATH_LOADERS + "valid_loader.pkl", 'rb') as valid_loader_pkl:
#     valid_loader = pickle.load(valid_loader_pkl)

df_train = pd.read_csv(path_train_set)
df_val = pd.read_csv(path_val_set)
df_test = pd.read_csv(path_test_set)
train_dataset = WordsWithLabelDataset(df_train, c2ind, w2ind)
valid_dataset = WordsWithLabelDataset(df_val, c2ind, w2ind)
test_dataset = WordsWithLabelDataset(df_test, c2ind, w2ind)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn,
                          num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn,
                          num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn,
                         num_workers=NUM_WORKERS)

if args.modeltype == 'lr':
    save_file = 'model_lr.pth'
    if args.train:
        # if USE_TOP50:
        #     model = BOWPool(len(ind2c_filtered), PATH_MY_EMBEDDINGS)
        # else:
        model = BOWPool(len(ind2c), PATH_MY_EMBEDDINGS)
    else:
        model = torch.load(os.path.join(PATH_OUTPUT, save_file))
elif args.modeltype == 'cnn':
    save_file = 'model_cnn.pth'
    if args.train:
        # if USE_TOP50:
        #     model = BOWPool(len(ind2c_filtered), PATH_MY_EMBEDDINGS)
        # else:
        model = BOWPool(len(ind2c), PATH_MY_EMBEDDINGS)
    else:
        model = torch.load(os.path.join(PATH_OUTPUT, save_file))
elif args.modeltype == 'rnn':
    save_file = 'model_rnn.pth'
    if args.train:
        # if USE_TOP50:
        #     model = BOWPool(len(ind2c_filtered), PATH_MY_EMBEDDINGS)
        # else:
        model = BOWPool(len(ind2c), PATH_MY_EMBEDDINGS)
    else:
        model = torch.load(os.path.join(PATH_OUTPUT, save_file))
elif args.modeltype == 'vanilla_cnn':
    save_file = 'model_vanilla_cnn.pth'
    if args.train:
        model = VanillaCNN(len(ind2c), PATH_MY_EMBEDDINGS)
    else:
        model = torch.load(os.path.join(PATH_OUTPUT, save_file))

if args.train:
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    criterion.to(device)
    best_val_acc = 0.0
    train_losses, train_mac_accs, train_mac_recs, train_mac_pres, train_mac_f1s, train_mac_aucs = [], [], [], [], [], []
    valid_losses, valid_mac_accs, valid_mac_recs, valid_mac_pres, valid_mac_f1s, valid_mac_aucs = [], [], [], [], [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_mac_acc, train_mac_rec, train_mac_pre, train_mac_f1, \
        train_mic_acc, train_mic_rec, train_mic_pre, train_mic_f1, train_mac_auc = train(model, device, train_loader,
                                                                                         criterion, optimizer, epoch,
                                                                                         verbose=False)
        valid_loss, valid_mac_acc, valid_mac_rec, valid_mac_pre, valid_mac_f1, \
        valid_mic_acc, valid_mic_rec, valid_mic_pre, valid_mic_f1, valid_mac_auc, _, _ = evaluate(model, device,
                                                                                                  valid_loader,
                                                                                                  criterion,
                                                                                                  verbose=False)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_mac_accs.append(train_mac_acc)
        valid_mac_accs.append(valid_mac_acc)

        train_mac_aucs.append(train_mac_auc)
        valid_mac_aucs.append(valid_mac_auc)

        # is_best = valid_mac_acc > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        is_best = valid_mac_auc > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        if is_best:
            # best_val_acc = valid_mac_acc
            best_val_acc = valid_mac_auc
            torch.save(model, os.path.join(PATH_OUTPUT, save_file))

    plot_learning_curves(train_losses, valid_losses, train_mac_aucs, valid_mac_aucs)

best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
print("Evaluating on test set")
test_loss, test_mac_acc, test_mac_rec, test_mac_pre, test_mac_f1, \
test_mic_acc, test_mic_rec, test_mic_pre, test_mic_f1, test_mac_auc, yhat_raws, y_all = evaluate(best_model, device,
                                                                                                 test_loader, criterion,
                                                                                                 verbose=False)

plot_roc_curve(yhat_raws, y_all)
