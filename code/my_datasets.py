from torch.utils.data import TensorDataset, Dataset
import numpy as np
import torch


# Reference: CSE6250 HW5
class WordsWithLabelDataset(Dataset):
    def __init__(self, df, c2ind_dict, w2ind_dict):
        self.num_labels = len(c2ind_dict)
        self.labels = []
        self.docs = []
        self.max_length = 2500  # max words in a document
        for index, row in df.iterrows():
            # Create labe; vector
            labels_idx = np.zeros(self.num_labels)
            found_at_least_one_label = False
            for l in row[3].split(';'):
                if l in c2ind_dict.keys():
                    code = int(c2ind_dict[l])
                    labels_idx[code] = 1
                    found_at_least_one_label = True
            if not found_at_least_one_label:
                continue
            # Add labels_idx to labels
            self.labels.append(labels_idx)
            # Convert words to indexes. If a word is not in embeddings it is considered unknown and the corresponding it is length of embedding words +1.
            # This index in the embeddings weight matrix is a random vector (Gaussian)
            text = [int(w2ind_dict[w]) if w in w2ind_dict else len(w2ind_dict) + 1 for w in row[2].split()]
            # Trancate doc
            if len(text) > self.max_length:
                text = text[:self.max_length]
            # Add docs to text
            text = np.array(text).astype(np.int)
            self.docs.append(text)

        # Convert labels to numpy
        self.labels = np.array(self.labels).astype(np.int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.docs[index], self.labels[index]


# Reference: CSE6250 HW5
def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(doc1, labels-one-hot encoded), (doc2, labels-one-hot encoded), ... , (docN, labelsN-one-hot encoded)]
    where N is minibatch size, doc is a (Sparse)FloatTensor, and label is a LongTensor

    :returns
        seqs (LongTensor) - 2D of batch_size X max doc length of batch
        labels (LongTensor) - 2D of batch_size. Each row represents a doc. Each column represents a label.
    """
    batch_size = len(batch)
    indices = np.arange(batch_size)
    doc_max_length = batch[0][0].size
    labels_length = batch[0][1].size
    lengths = np.zeros(batch_size).astype(int)

    for i, (doc, labels) in enumerate(batch):
        length = doc.size
        lengths[i] = length
        if length > doc_max_length:
            doc_max_length = length

    sorted_indices = [x for _, x in sorted(zip(lengths, indices), reverse=True)]
    sorted_labels = np.zeros((batch_size, labels_length), dtype=int)
    sorted_docs = np.zeros((batch_size, doc_max_length), dtype=int)

    for i, indice in enumerate(sorted_indices):
        element = batch[indice]
        doc = element[0]
        doc_length = doc.size
        label = element[1]
        sorted_labels[i] = label
        sorted_docs[i, 0:doc_length] = doc[np.newaxis, :]

    docs_tensor = torch.LongTensor(sorted_docs)
    labels_tensor = torch.FloatTensor(sorted_labels)

    return docs_tensor, labels_tensor
