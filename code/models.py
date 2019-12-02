import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable


class BaseModel(nn.Module):

    def __init__(self, Y, embed_file):
        super(BaseModel, self).__init__()
        self.Y = Y  # label size

        # make embedding layer
        print("loading pretrained embeddings 1...")
        w = torch.Tensor(load_embeddings(embed_file))
        num_embeddings, embedding_dim = w.size()  # num_embeddings words in vocab, embedding_dim dimensional embeddings
        self.embed_size = embedding_dim

        self.embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.embed.weight.data = w.clone()
        # Make embedding weights non trainable
        self.embed.weight.requires_grad = True


class BOWPool(BaseModel):
    """
        Logistic regression model over average or max-pooled word vector input
    """

    def __init__(self, Y, embed_file):
        super(BOWPool, self).__init__(Y, embed_file)
        self.final = nn.Linear(self.embed_size, Y)
        xavier_uniform_(self.final.weight)
        self.embed_drop = nn.Dropout(p=0.5)

    def forward(self, x):
        seq_length = x.shape[1]
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        x = nn.MaxPool1d(kernel_size=seq_length)(x)
        x = torch.squeeze(x, 2)
        x = self.final(x)
        return x