import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable
from math import floor
from config import *


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


class VanillaCNN(BaseModel):
    def __init__(self, Y, embed_file):
        num_filters = 100
        kernel_size = 4
        super(VanillaCNN, self).__init__(Y, embed_file)
        self.conv = nn.Conv1d(self.embed_size, num_filters, kernel_size)
        xavier_uniform_(self.conv.weight)
        self.embed_drop = nn.Dropout(p=0.5)

        self.final = nn.Linear(num_filters, Y)
        xavier_uniform_(self.final.weight)

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = nn.MaxPool1d(x.size()[2])(torch.tanh(x))
        x = torch.squeeze(x, 2)
        # linear output
        x = self.final(x)
        return x


class BidirectionalGru(BaseModel):
    def __init__(self, Y, embed_file):
        super(BidirectionalGru, self).__init__(Y, embed_file)
        self.rnn_dim = 512
        self.num_directions = 2
        self.num_layers = 1
        self.rnn = nn.GRU(self.embed_size,
                          hidden_size=floor(self.rnn_dim / self.num_directions),
                          num_layers=self.num_layers,
                          bidirectional=True)
        self.final = nn.Linear(self.rnn_dim, Y)

    def forward(self, x):
        embeds = self.embed(x).transpose(0, 1)
        out, hidden = self.rnn(embeds)
        batch_size_current = hidden.shape[1]
        last_hidden = hidden[-2:].transpose(0, 1).contiguous().view(batch_size_current, -1)
        # print(last_hidden.size())
        yhat = self.final(last_hidden)
        return yhat


class CNNAttn(BaseModel):
    def __init__(self, Y, embed_file):
        num_filters = 50
        kernel_size = 10
        super(CNNAttn, self).__init__(Y, embed_file)
        self.conv = nn.Conv1d(self.embed_size, num_filters, kernel_size, padding=int(floor(kernel_size / 2)))
        xavier_uniform_(self.conv.weight)
        self.embed_drop = nn.Dropout(p=0.5)

        self.U = nn.Linear(num_filters, Y)
        xavier_uniform_(self.U.weight)

        self.final = nn.Linear(num_filters, Y)
        xavier_uniform_(self.final.weight)

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        # apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return y
