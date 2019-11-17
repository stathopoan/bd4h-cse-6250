import torch
from torch.autograd import Variable

def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():
        one_hot_output = output > 0.5
        correct = torch.sum(one_hot_output == target)
        batch_size = target.size(0)
        # _, pred = output.max(1)
        # correct = pred.eq(target).sum()
        acc = (correct * 100.0) / (target.shape[0]*target.shape[1])
        # return correct * 100.0 / batch_size
        return acc



A = torch.tensor([[0.345, 0.573, 0.22], [0.213, 0.778, 0.34]])
B = torch.tensor([[1, 1, 0], [0, 1, 1]])

A_ = A >0.5

acc = compute_batch_accuracy(A_, B)

# E = torch.eq(A_ ,B)
# x_A__ = Variable(A_)
# x_A = Variable(A)
# x_B = Variable(B)
# x_E = Variable(E)
#
# correct = torch.sum(A_  == B)

# print (x_A)
# print (x_B)
# print (x_A__ )
# print(correct)

print (acc)

