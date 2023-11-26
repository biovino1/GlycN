"""Defines 1D-CNN model for predicting glyscosylation.

Inputs: 1D tensors of length 2560 (ESM-2 t36)

__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """1D CNN block with 1D convolution, batch normalization, ReLU activation, and flattening for
    input to feed forward block.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1d(x))
        x = self.bn(x)
        x = self.flatten(x)
        return x


class FeedForwardBlock(nn.Module):
    """Feed forward block with two dense layers and sigmoid activation. Takes input from CNN block.
    """

    def __init__(self, in_features, hidden_dim, out_features, dropout):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x


class GlycN(nn.Module):
    """1D CNN with feed forward block.
    """

    def __init__(self, config):
        super().__init__()
        self.conv_block = ConvBlock(**config['ConvBlock'])
        self.ff_block = FeedForwardBlock(**config['FeedForwardBlock'])
        self.model_params = config['GlycN']


    def forward(self, x):
        x = self.conv_block(x)
        x = self.ff_block(x)
        return x
