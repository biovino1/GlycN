"""Defines 1D-CNN model for predicting glyscosylation.


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""


import torch.nn as nn
import torch.optim as optim


class ConvBlock(nn.Module):
    """1D CNN block with 1D convolution, batch normalization, ReLU activation, and flattening for
    input to feed forward block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
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
        x = self.linear2(x)
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


    def train_model(self, X_train, X_test, y_train, y_test):
        """Trains model based on model_params defined in config.ini.

        :param X_train: array of 1xn embeddings for testing
        :param X_test: array of 1xn embeddings for testing
        :param y_train: array of labels for training
        :param y_test: array of labels for testing
        """

        loss_fxn = self.model_params['loss']
        optimizer = optim.Adam(self.parameters(), lr=self.model_params['lr'])

        # Training loop
        for epoch in range(self.model_params['epochs']):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = loss_fxn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            # Print loss
            if epoch % 10 == 0:
                print(f'Epoch: {epoch} | Loss: {loss.item()}')

        # Evaluate model
        self.eval()
