"""Trains model defined on model.py using data from N_embeds.npy


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import yaml
from model import GlycN
from embed import PytorchDataset
from sklearn.metrics import roc_curve, auc

torch.cuda.set_per_process_memory_fraction(0.8)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def conf_mat(outputs: list, labels: list) -> tuple:
    """Returns confusion matrix for given outputs and labels.
    
    :param outputs: list of model outputs
    :param labels: list of labels
    :return tuple: confusion matrix (tn, fp, fn, tp)
    """

    tn, fp, fn, tp = 0, 0, 0, 0
    for i, output in enumerate(outputs):
        if output == 0 and labels[i] == 0:
            tn += 1
        elif output == 1 and labels[i] == 0:
            fp += 1
        elif output == 0 and labels[i] == 1:
            fn += 1
        elif output == 1 and labels[i] == 1:
            tp += 1

    return (tn, fp, fn, tp)


def get_metrics(outputs: list, labels: list) -> tuple:
    """Returns accuracy, precision, recall, F1, AUC, MCC for given outputs and labels.

    :param outputs: list of model outputs
    :param labels: list of labels
    :return tuple: accuracy, precision, recall, F1, AUC, MCC
    """

    # Calculate metrics
    tn, fp, fn, tp = conf_mat(outputs, labels)
    acc = (tp + tn) / len(outputs)
    prec = tp / (tp + fp) if tp + fp != 0 else 0
    rec = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0
    fpr, tpr, _ = roc_curve(labels, outputs, pos_label=1)
    aucs = auc(fpr, tpr)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return (acc, prec, rec, f1, aucs, mcc)


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


def train_model(config: dict, train_loader: DataLoader, device: str) -> GlycN:
    """Returns trained model.

    :param config: dict with model parameters
    :param train_loader: pytorch DataLoader
    :param device: device to train on
    :return GlycN: trained model
    """

    model = GlycN(config)
    model.to(device)

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

    return model


def predict(
        model: GlycN, embeds_test: torch.Tensor, labels_test: torch.Tensor, criterion) -> tuple:
    """Returns accuracy and loss of model on validation data.

    :param model: GlycN model
    :param embeds_val: validation data
    :param labels_val: validation labels
    :param criterion: loss function
    :return tuple: rounded outputs and loss
    """

    model.eval()
    with torch.no_grad():
        outputs = model(embeds_test).flatten()
        outputs = torch.round(outputs)
        loss = criterion(outputs, labels_test)
    torch.cuda.empty_cache()

    return outputs, loss


def report_scores(scores: tuple):
    """Prints scores from model evaluation on testing data.
    """

    print('Scores from model evaluation on testing data:')
    print('acc: {:.3f}, prec: {:.3f}, rec: {:.3f}, f1: ' \
          '{:.3f}, auc: {:.3f}, mcc: {:.3f}'.format(*scores))  #pylint: disable=C0209


def main():
    """Main function
    """

    if not os.path.exists('data/models'):
        os.makedirs('data/models')

    # Load config and get device
    with open('scripts/config.yaml', 'r', encoding='utf8') as cfile:
        config = yaml.safe_load(cfile)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get training data and train model
    train_loader = get_train_loader(config)
    model = train_model(config, train_loader, device)

    # Get testing data and evaluate model
    embeds_test = torch.load('data/datasets/embeds_test.pt').to(device)
    labels_test = torch.load('data/datasets/labels_test.pt').to(device).float()
    outputs, _ = predict(model, embeds_test, labels_test, nn.BCELoss())
    outputs, labels_test = outputs.cpu(), labels_test.cpu()
    scores = get_metrics(outputs, labels_test)
    report_scores(scores)

    # Save model
    torch.save(model.state_dict(), 'data/models/model.pt')


if __name__ == '__main__':
    main()
