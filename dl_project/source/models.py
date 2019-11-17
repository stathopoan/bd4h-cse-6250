import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file):
        super(BaseModel, self).__init__()
        self.Y = Y # label size

        # make embedding layer
        print("loading pretrained embeddings...")
        w = torch.Tensor(load_embeddings(embed_file))
        num_embeddings, embedding_dim = w.size() # num_embeddings words in vocab, embedding_dim dimensional embeddings
        self.embed_size = embedding_dim

        self.embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.embed.weight.data = w.clone()
        # Make embedding weights non trainable
        self.embed.weight.requires_grad = False

class BOWPool(BaseModel):
    """
        Logistic regression model over average or max-pooled word vector input
    """
    def __init__(self, Y, embed_file):
        super(BOWPool, self).__init__(Y, embed_file)
        self.final = nn.Linear(self.embed_size, Y)
        # xavier_uniform(self.final.weight)
        # self.pool = nn.MaxPool1d(2500)

    def forward(self, x):
        seq_length = x.shape[1]
        x = self.embed(x)
        # x = nn.Dropout(p=0.5)
        x = x.transpose(1, 2)
        x = nn.AvgPool1d(seq_length)(x)
        x = torch.squeeze(x)
        logits = torch.sigmoid(self.final(x))

        return logits