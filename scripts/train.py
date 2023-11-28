"""Trains model defined on model.py using data from N_embeds.npy


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import yaml
from model import GlycN
from embed import PytorchDataset

torch.cuda.set_per_process_memory_fraction(0.8)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


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


def get_acc_loss(
        model: GlycN, embeds_test: torch.Tensor, labels_test: torch.Tensor, criterion) -> tuple:
    """Returns accuracy and loss of model on validation data.

    :param model: GlycN model
    :param embeds_val: validation data
    :param labels_val: validation labels
    :param criterion: loss function
    :return tuple: percent accuracy and loss
    """

    model.eval()
    with torch.no_grad():
        outputs = model(embeds_test).flatten()
        outputs = torch.round(outputs)
        total = labels_test.size(0)
        correct = (outputs == labels_test).sum().item()
        loss = criterion(outputs, labels_test)
    torch.cuda.empty_cache()

    return correct / total, loss


def main():
    """Main function
    """

    if not os.path.exists('data/models'):
        os.makedirs('data/models')

    # Load model with config file parameters
    with open('scripts/config.yaml', 'r', encoding='utf8') as cfile:
        config = yaml.safe_load(cfile)
    model = GlycN(config)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Get training and validation data
    train_loader = get_train_loader(config)
    embeds_test = torch.load('data/datasets/embeds_test.pt').to(device)
    labels_test = torch.load('data/datasets/labels_test.pt').to(device).float()

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['GlycN']['lr'])

    # Training loop
    for _ in range(config['GlycN']['epochs']):
        model.train()
        for batch in train_loader:
            data = batch['embed'].to(device)
            labels = batch['label'].to(device).float()

            # Get outputs and calculate loss
            outputs = model(data).flatten()
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Print loss and accuracy on validation data
    acc, loss = get_acc_loss(model, embeds_test, labels_test, criterion)
    print(f'Accuracy: {round((100 * acc), 2)}%')


if __name__ == '__main__':
    main()