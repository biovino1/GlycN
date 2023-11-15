"""Trains model defined on model.py using data from N_embeds.npy


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import yaml
from model import GlycN
from embed import PytorchDataset


def get_train_loader(config: dict) -> DataLoader:
    """Returns pytorch DataLoader for training data.

    :param config: dict with model parameters
    :return: pytorch DataLoader
    """

    # Load training data
    embeds_train = torch.load('data/datasets/embeds_train.pt')
    labels_train = torch.load('data/datasets/labels_train.pt')

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
    train_loader = get_train_loader(config)

    # Send model and data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['GlycN']['lr'])

    # Training loop
    model.train()
    for epoch in range(config['GlycN']['epochs']):
        print(f'Epoch: {epoch}')

        # Keep track of accuracy each epoch
        correct = 0
        total = 0
        for batch in train_loader:
            data = batch['embed'].to(device)
            labels = batch['label'].to(device).float()

            # Get outputs and convert sigmoid output to binary prediction
            outputs = model(data)
            outputs = torch.round(outputs).flatten()

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate accuracy
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print or log the training loss for this epoch if needed
        print(f'Epoch [{epoch + 1}/{config["GlycN"]["epochs"]}], Loss: {loss.item()}')
        print(f'Accuracy: {round((100 * correct / total), 2)}%')


if __name__ == '__main__':
    main()
