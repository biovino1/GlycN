"""Trains model defined on model.py using data from N_embeds.npy


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from embed import GlycDataset, PytorchDataset
from model import GlycN
import yaml


def get_train_loader(embeds_path: str, config: dict):
    """Returns pytorch DataLoader for training data.

    :param embeds_path: path to N_embeds.npy
    :param config: dict with model parameters
    :return: pytorch DataLoader
    """

    # Load data
    dataset = GlycDataset(embeds_path)
    dataset.get_data()
    embeds_train, _, labels_train, _ = dataset.split(0.2)

    # Define pytorch dataset
    batch_size = config['GlycN']['batch_size']
    train_dataset = PytorchDataset(embeds_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def main():
    """Main function
    """

    # Define model parameters
    with open('scripts/config.yaml', 'r', encoding='utf8') as cfile:
        config = yaml.safe_load(cfile)

    # Train model
    model = GlycN(config)
    train_loader = get_train_loader('data/N_embeds.npy', config)

    # Send model and data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for batch in train_loader:
        batch['embed'] = batch['embed'].to(device)
        batch['label'] = batch['label'].to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['GlycN']['lr'])

    # Training loop
    for epoch in range(config['GlycN']['epochs']):
        print(f'Epoch: {epoch}')
        model.train()
        for batch in train_loader:
            data = batch['embed']
            labels = batch['label']

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass and loss calculation
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        # Print or log the training loss for this epoch if needed
        print(f'Epoch [{epoch + 1}/{10}], Loss: {loss.item()}')


if __name__ == '__main__':
    main()
