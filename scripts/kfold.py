"""Performs k-fold cross validation on GlycN model.


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

import argparse
import logging
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import yaml
from model import GlycN
from embed import PytorchDataset
from sklearn.model_selection import KFold
from train import train_model, get_metrics

log_filename = 'data/logs/kfold.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')

torch.cuda.set_per_process_memory_fraction(0.8)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def validate(
        model: GlycN, val_data: DataLoader, criterion, device: str) -> tuple:
    """Returns several metrics for given model and validation data.

    :param model: GlycN model
    :param val_data: validation data
    :param criterion: loss function
    :param device: device for training
    :return tuple: accuracy, precision, recall, F1, AUC, MCC, loss
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

    return scores + (loss.item(),)


def report_results(results: dict):
    """Prints results from k-fold cross validation.

    :param results: dictionary of results from k-fold cross validation
    """

    # Add values from fold to metrics dict (combination of all folds)
    metrics = {}
    for fold in results:
        values = []
        for metric in results[fold]:
            values.append(results[fold][metric])
            metrics[metric] = metrics.get(metric, []) + [results[fold][metric]]

        # Print results for fold
        logging.info('Fold {}: acc: {:.3f}, prec: {:.3f}, rec: {:.3f}, f1: {:.3f}, '  #pylint: disable=C0209
                     'auc: {:.3f}, mcc: {:.3f}, loss: {:.3f}'.format(fold, *values))
    logging.info('')

    # Calculate averages and standard deviations
    for metric in metrics:
        metrics[metric] = (np.mean(metrics[metric]), np.std(metrics[metric]))

    # Print results
    logging.info('Results from k-fold cross validation:')
    for metric, values in metrics.items():
        logging.info('{}: {:.3f} ± {:.3f}'.format(metric, values[0], values[1]))  #pylint: disable=C0209
    logging.info('')


def main():
    """Main function
    """

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='scripts/config.yaml', help='config file')
    args = parser.parse_args()

    if not os.path.exists('data/models'):
        os.makedirs('data/models')

    # Define model parameters and device for training
    with open(args.c, 'r', encoding='utf8') as cfile:
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

        # Train model
        model = train_model(config, train_loader, device)

        # Evaluate on validation data for this fold
        criterion = nn.BCELoss()
        scores = validate(model, val_loader, criterion, device)
        results[fold] = {'acc': scores[0], 'prec': scores[1], 'rec': scores[2],
                        'f1': scores[3], 'auc': scores[4], 'mcc': scores[5], 'loss': scores[6]}

    report_results(results)


if __name__ == '__main__':
    main()
