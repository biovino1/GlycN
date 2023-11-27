"""Trains model defined on model.py using data from N_embeds.npy


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

import logging
import numpy as np
import os
import sys
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import yaml
from model import GlycN
from embed import PytorchDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

torch.cuda.set_per_process_memory_fraction(0.8)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def get_metrics(outputs: list, labels: list) -> tuple:
    """Returns accuracy, precision, recall, F1, AUC, MCC for given outputs and labels.

    :param outputs: list of model outputs
    :param labels: list of labels
    :return tuple: accuracy, precision, recall, F1, AUC, MCC
    """

    # Get TN, FP, FN, TP
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

    # Calculate metrics
    acc = (tp + tn) / len(outputs)
    prec = tp / (tp + fp) if tp + fp != 0 else 0
    rec = tp / (tp + fn) if tp + fn != 0 else 0
    fpr = fp / (fp + tn) if fp + tn != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0
    aucs = auc(fpr, rec)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return (acc, prec, rec, f1, aucs, mcc)


def validate(
        model: GlycN, val_data: DataLoader, criterion, device: str) -> tuple:
    """Returns several metrics for given model and validation data.

    :param model: GlycN model
    :param val_data: validation data
    :param criterion: loss function
    :param device: device for training
    :return tuple: loss, accuracy, precision, recall, F1, AUC, MCC
    """

    model.eval()
    total_outputs, total_labels = [], []
    with torch.no_grad():
        for batch in val_data:
            embeds = batch['embed'].to(device)
            labels = batch['label'].to(device).float()

            # Get outputs and calculate loss
            outputs = model(embeds).flatten()
            loss = criterion(outputs, labels)

            # Add outputs and labels to lists for calculating metrics
            outputs = torch.round(outputs)
            total_outputs.extend(outputs.tolist())
            total_labels.extend(labels.tolist())

    # Calculate metrics
    scores = get_metrics(total_outputs, total_labels)

    return scores, loss.item()


def report_results(results: dict):
    """Prints results from k-fold cross validation.

    :param results: dictionary of results from k-fold cross validation
    """

    # Calculate average and standard deviation for each metric
    metrics = {}
    for fold in results:

        # Print results for fold
        logging.info('Fold {}: acc: {:.3f}, prec: {:.3f}, rec: {:.3f}, f1: {:.3f}, loss: {:.3f}'.format(
            fold, results[fold]['acc'], results[fold]['prec'], results[fold]['rec'], results[fold]['f1'], results[fold]['loss']))

        for metric in results[fold]:
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(results[fold][metric])
    for metric in metrics:
        metrics[metric] = (np.mean(metrics[metric]), np.std(metrics[metric]))

    # Print averages and stds
    logging.info('Results from k-fold cross validation:')
    logging.info('acc: {:.3f} ± {:.3f}'.format(metrics['acc'][0], metrics['acc'][1]))
    logging.info('prec: {:.3f} ± {:.3f}'.format(metrics['prec'][0], metrics['prec'][1]))
    logging.info('rec: {:.3f} ± {:.3f}'.format(metrics['rec'][0], metrics['rec'][1]))
    logging.info('f1: {:.3f} ± {:.3f}'.format(metrics['f1'][0], metrics['f1'][1]))
    logging.info('loss: {:.3f} ± {:.3f}'.format(metrics['loss'][0], metrics['loss'][1]))
    logging.info('')


def main():
    """Main function
    """

    if not os.path.exists('data/models'):
        os.makedirs('data/models')

    # Define model parameters and device for training
    with open('scripts/config.yaml', 'r', encoding='utf8') as cfile:
        config = yaml.safe_load(cfile)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get training data
    embeds_train = torch.load('data/datasets/embeds_train.pt')
    labels_train = torch.load('data/datasets/labels_train.pt')
    train_dataset = PytorchDataset(embeds_train, labels_train)

    # K-fold cross validation
    results = {}
    kfold = KFold(n_splits=5, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):

        # Sample elements randomly from training and validation splits
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        # Define data loader for this fold
        batch_size = config['GlycN']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler)

        # Define model
        model = GlycN(config)
        model.to(device)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['GlycN']['lr'])

        # Training loop
        for _ in range(10):
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

        # Evaluate on validation data for this fold
        scores, loss = validate(model, val_loader, criterion, device)
        results[fold] = {'acc': scores[0], 'prec': scores[1], 'rec': scores[2],
                        'f1': scores[3], 'auc': scores[4], 'mcc': scores[5], 'loss': loss}

    report_results(results)


if __name__ == '__main__':
    main()
