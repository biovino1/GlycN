"""Defines 1D-CNN model for predicting glyscosylation.


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

from torch import nn


class ConvBlock(nn.Module):
    """1D CNN block with 1D convolution, batch normalization, ReLU activation, and flattening for
    input to feed forward block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_kernel)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class FeedForwardBlock(nn.Module):
    """Feed forward block with two dense layers and sigmoid activation. Takes input from CNN block.
    """

    def __init__(self, in_features, hidden_dim, out_features, dropout):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
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
