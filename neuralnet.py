import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, in_dim=30, out_dim=1, hidden_dims=None):
        super(NeuralNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [120, 24]

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim

        d_ = in_dim
        layers = []
        for d in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(d_, d),
                    nn.BatchNorm1d(d),
                    nn.ReLU())
                )
            d_ = d
        layers.append(nn.Linear(d_, out_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def loss(self, y_, y, zero_weight=0.1):
        ones = torch.nonzero(y)
        zeros = torch.nonzero(y-1)
        return torch.mean(torch.square(y[ones]-y_[ones])) + zero_weight * torch.mean(torch.square(y[zeros]-y_[zeros]))
#       return F.binary_cross_entropy(y_[ones], y[ones]) + zero_weight * F.binary_cross_entropy(y_[zeros], y[zeros])
