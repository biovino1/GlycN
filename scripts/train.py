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
        model: GlycN, embeds_val: torch.Tensor, labels_val: torch.Tensor, criterion) -> tuple:
    """Returns accuracy and loss of model on validation data.

    :param model: GlycN model
    :param embeds_val: validation data
    :param labels_val: validation labels
    :param criterion: loss function
    :return tuple: percent accuracy and loss
    """

    model.eval()
    with torch.no_grad():
        outputs = model(embeds_val).flatten()
        outputs = torch.round(outputs)
        total = labels_val.size(0)
        correct = (outputs == labels_val).sum().item()
        loss = criterion(outputs, labels_val)
    torch.cuda.empty_cache()

    return correct / total, loss


def early_stop(acc: float, acc_prev: float, min_delta: float, patience: int) -> bool:
    """Returns True if training should stop.

    :param acc: current accuracy
    :param acc_prev: previous accuracy
    :param min_delta: minimum change in accuracy to be considered improvement
    :param patience: number of epochs to wait before stopping
    :return bool: True if training should stop, False o.w.
    """

    if acc - acc_prev > min_delta:
        return patience
    return patience - 1


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
    embeds_val = torch.load('data/datasets/embeds_train.pt').to(device)
    labels_val = torch.load('data/datasets/labels_train.pt').to(device).float()

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['GlycN']['lr'])

    # Training loop
    patience, acc_prev, best_acc = config['GlycN']['patience'], 0, 0
    for epoch in range(config['GlycN']['epochs']):
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
        acc, loss = get_acc_loss(model, embeds_val, labels_val, criterion)
        print(f'Epoch [{epoch + 1}/{config["GlycN"]["epochs"]}]')
        print(f'Loss: {loss}')
        print(f'Accuracy: {round((100 * acc), 2)}%')

        # Save model if accuracy improved
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'data/models/GlycN.pt')

        # Stop training if accuracy not improving
        patience = early_stop(acc, acc_prev, config['GlycN']['min_delta'], patience)
        if patience == 0:
            print('Early Stopping')
            break
        acc_prev = acc


if __name__ == '__main__':
    main()
