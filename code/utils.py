import numpy as np
import pandas as pd
from collections import defaultdict
import csv
import os
import time
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

from plots import plot_roc_curve

DESC_CODES_PATH = "../data"


def load_embeddings(embed_file):
    """
    Load the embeddings from the file trained already. Also normalize them.
    Add one vector for padding (full of zeros)
    Add one vector for UNK token (random vector)
    :param embed_file: The embeddings file trained containing the word and the vectors
    :return: a numpy matrix with only the vectors normalized
    """
    df = pd.read_csv(embed_file)
    df_vectors = df["vector"].str.split(" ")
    W = np.array(df_vectors.values.tolist()).astype(np.float)
    # Create a row to represent pad vector (all zeros)
    padrow = np.array(np.zeros(len(W[-1]))).astype(np.float)
    # Place it at first position
    W = np.insert(W, 0, padrow, axis=0)
    # Create a row to represent UNK token vector (random: Gaussian)
    unkrow = np.array(np.random.randn(len(W[-1]))).astype(np.float)
    # Place it last
    W = np.vstack((W, unkrow))
    # Normalize every row and add a small number to avoid division by zero
    norm2 = (np.linalg.norm(W, axis=1) + 1e-6).astype(np.float)
    W = W / norm2[:, np.newaxis]
    return W


def get_words_to_indexes_dictionaries(embed_file):
    """
    Create a vocabulary from the embeddings file and for every word create and corresponding index and vice versa
    :param embed_file: The file with the embeddings words
    :return: Two dictionaries
    """
    df = pd.read_csv(embed_file)
    ind2w = {i + 1: w[0] for i, w in df[["word"]].iterrows()}
    w2ind = {w: i for i, w in ind2w.items()}  # Start from 1 index. 0 index will represent the padding weights
    return ind2w, w2ind


# Reference: https://github.com/jamesmullenbach/caml-mimic/blob/master
def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


# Reference: https://github.com/jamesmullenbach/caml-mimic/blob/master
def load_code_descriptions(data_dir):
    # load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    with open("%s/D_ICD_DIAGNOSES.csv" % (data_dir), 'r') as descfile:
        r = csv.reader(descfile)
        # header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            desc_dict[reformat(code, True)] = desc
    with open("%s/D_ICD_PROCEDURES.csv" % (data_dir), 'r') as descfile:
        r = csv.reader(descfile)
        # header
        next(r)
        for row in r:
            code = row[1]
            desc = row[-1]
            if code not in desc_dict.keys():
                desc_dict[reformat(code, False)] = desc
    with open('%s/ICD9_descriptions' % data_dir, 'r') as labelfile:
        for i, row in enumerate(labelfile):
            row = row.rstrip().split()
            code = row[0]
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[1:])
    return desc_dict


# Reference: https://github.com/jamesmullenbach/caml-mimic/blob/master
def load_full_codes(train_path):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup, description lookup
    """
    # get description lookup
    desc_dict = load_code_descriptions(DESC_CODES_PATH)
    # build code lookups from appropriate datasets

    codes = set()
    for split in ['train', 'val', 'test']:  # Expect 3 files with the only difference to be the train/
        with open(train_path.replace('train', split), 'r') as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                for code in row[3].split(';'):
                    codes.add(code)
    codes = set([c for c in codes if c != ''])
    ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    c2ind = {c: i for i, c in ind2c.items()}
    return ind2c, c2ind, desc_dict


# Reference: CSE6250 HW5
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Reference: https://github.com/jamesmullenbach/caml-mimic/
def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


# Reference: https://github.com/jamesmullenbach/caml-mimic/
def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def inst_precision(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    # correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def inst_recall(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    # correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)


# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def inst_f1(yhat, y):
    prec = inst_precision(yhat, y)
    rec = inst_recall(yhat, y)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

# Reference: https://github.com/jamesmullenbach/caml-mimic/evaluation.py
def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc


# Reference: CSE6250 HW5
def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10, verbose=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    y_hats = []
    y_all = []
    yhat_raws = []

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        y = target.cpu().detach().numpy()
        yhat = torch.sigmoid(output).cpu().detach().round().numpy()
        yhat_raw = torch.sigmoid(output).cpu().detach().numpy()
        y_all.append(y)
        y_hats.append(yhat)
        yhat_raws.append(yhat_raw)
        f1 = inst_f1(yhat, y)

        if i % print_freq == 0 and verbose:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'F1 {f1:.3f} \t'.format(epoch,
                                          i,
                                          len(data_loader),
                                          batch_time=batch_time,
                                          data_time=data_time,
                                          loss=losses,
                                          f1=f1))

        # if i == 1239:
        #     print(True)

    y_hats = np.concatenate(y_hats, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    yhat_raws = np.concatenate(yhat_raws, axis=0)

    mac_acc = macro_accuracy(y_hats, y_all)
    mac_rec = macro_recall(y_hats, y_all)
    mac_pre = macro_precision(y_hats, y_all)
    mic_acc = micro_accuracy(y_hats.ravel(), y_all.ravel())
    mic_rec = micro_recall(y_hats.ravel(), y_all.ravel())
    mic_pre = micro_precision(y_hats.ravel(), y_all.ravel())
    mic_f1 = micro_f1(y_hats.ravel(), y_all.ravel())
    mac_f1 = macro_f1(y_hats, y_all)
    auc_dict = auc_metrics(yhat_raws, y_all, y_all.ravel())

    print('Epoch: [{0}] \t'
          'Train: [{1}] \t'
          'Time {batch_time.avg:.3f}\t'
          'Data {data_time.avg:.3f}\t'
          'Loss {loss.avg:.4f} \t'
          'Macro Accuracy {mac_acc:.3f} \t'
          'Macro Recall {mac_rec:.3f} \t'
          'Macro Precision {mac_pre:.3f} \t'
          'Macro F1 {mac_f1:.3f} \t'
          'Macro AUC {mac_auc:.3f} \t'
          'Micro Accuracy {mic_acc:.3f} \t'
          'Micro Recall {mic_rec:.3f} \t'
          'Micro Precision {mic_pre:.3f} \t'
          'Micro F1 {mic_f1:.3f} \t'
          'Micro AUC {auc_micro:.3f} \t'.format(
        epoch,
        len(data_loader),
        batch_time=batch_time,
        data_time=data_time,
        loss=losses,
        mac_acc=mac_acc,
        mac_pre=mac_pre,
        mac_rec=mac_rec,
        mac_f1=mac_f1,
        mac_auc=auc_dict['auc_macro'],
        mic_acc=mic_acc,
        mic_pre=mic_pre,
        mic_rec=mic_rec,
        mic_f1=mic_f1,
        auc_micro=auc_dict['auc_micro']))

    return losses.avg, mac_acc, mac_pre, mac_rec, mac_f1, mic_acc, mic_pre, mic_rec, mic_f1, auc_dict['auc_macro']


# Reference: CSE6250 HW5
def evaluate(model, device, data_loader, criterion, print_freq=10, verbose=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    y_hats = []
    y_all = []
    yhat_raws = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            y = target.cpu().detach().numpy()
            y_all.append(y)
            yhat = torch.sigmoid(output).cpu().detach().round().numpy()
            y_hats.append(yhat)
            yhat_raw = torch.sigmoid(output).cpu().detach().numpy()
            yhat_raws.append(yhat_raw)

            if i % print_freq == 0 and verbose:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(data_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time,
                                                                      loss=losses,
                                                                      ))
    y_hats = np.concatenate(y_hats, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    yhat_raws = np.concatenate(yhat_raws, axis=0)

    mac_acc = macro_accuracy(y_hats, y_all)
    mac_rec = macro_recall(y_hats, y_all)
    mac_pre = macro_precision(y_hats, y_all)
    mic_acc = micro_accuracy(y_hats.ravel(), y_all.ravel())
    mic_rec = micro_recall(y_hats.ravel(), y_all.ravel())
    mic_pre = micro_precision(y_hats.ravel(), y_all.ravel())
    mic_f1 = micro_f1(y_hats.ravel(), y_all.ravel())
    mac_f1 = macro_f1(y_hats, y_all)
    auc_dict = auc_metrics(yhat_raws, y_all, y_all.ravel())

    print('Test: \t'
          'Loss {loss.avg:.4f} \t'
          'Macro Accuracy {mac_acc:.3f} \t'
          'Macro Recall {mac_rec:.3f} \t'
          'Macro Precision {mac_pre:.3f} \t'
          'Macro F1 {mac_f1:.3f} \t'
          'Macro AUC {mac_auc:.3f} \t'
          'Micro Accuracy {mic_acc:.3f} \t'
          'Micro Recall {mic_rec:.3f} \t'
          'Micro Precision {mic_pre:.3f} \t'
          'Micro F1 {mic_f1:.3f} \t'
          'Micro AUC {auc_micro:.3f} \t'.format(
        len(data_loader),
        data_time=data_time,
        loss=losses,
        mac_acc=mac_acc,
        mac_pre=mac_pre,
        mac_rec=mac_rec,
        mac_f1=mac_f1,
        mac_auc = auc_dict['auc_macro'],
        mic_acc=mic_acc,
        mic_pre=mic_pre,
        mic_rec=mic_rec,
        mic_f1=mic_f1,
        auc_micro=auc_dict['auc_micro']))

    return losses.avg, mac_acc, mac_pre, mac_rec, mac_f1, mic_acc, mic_pre, mic_rec, mic_f1, auc_dict['auc_macro'], yhat_raws, y_all


def load_frequency_weights(frequency_labels_path, c2ind):
    df = pd.read_csv(frequency_labels_path, header=None, names=['label', 'frequency'])
    indices = [c2ind[row[0]] for i, row in df.iterrows()]
    df['indices'] = indices
    df = df.sort_values(by=['indices'], ascending=True)
    weights = [r[1] for i, r in df.iterrows()]
    w = np.array(weights, dtype=np.float32)
    w = 1.0 - w / np.sum(w)
    return w
