"""Performs grid search on GlycN model.

__author__ = "Ben Iovino"
__date__ = "12/06/23"
"""

import datetime
import logging
import os
from itertools import product
import yaml

log_filename = 'data/logs/grid_search.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='a',
                     level=logging.INFO, format='%(message)s')


def define_grid() -> dict:
    """Returns a dict of hyperparameters to test.

    :return dict: hyperparameters
    """

    grid = {
        'out_channels': [8, 16, 32, 64, 128],
        'kernel_size': [1, 2, 3, 4, 5],
        'hidden_dim': [8, 16, 32, 64, 128],
        'lr': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'epochs': [10, 20, 30, 40, 50],
        'batch_size': [8, 16, 32, 64, 128]
    }

    return grid


def write_config(grid: dict):
    """Writes grid search config file.

    :param grid: hyperparameters
    """

    # Separate first 3 hyperpameters into ConvBlock
    conv_block = {
        'in_channels': 1,
        'out_channels': grid['out_channels'],
        'kernel_size': grid['kernel_size']
    }

    # Separate next 4 hyperparameters into FeedForwardBlock
    feedforward_block = {
        'in_features': 2559*grid['out_channels'] - (grid['kernel_size'] - 2),
        'hidden_dim': grid['hidden_dim'],
        'out_features': 1,
        'dropout': 0.2
    }

    # Last 3 are GlycN
    glyc_n = {
        'lr': grid['lr'],
        'epochs': grid['epochs'],
        'batch_size': grid['batch_size']
    }

    # Combine all 3
    grid = {
        'ConvBlock': conv_block,
        'FeedForwardBlock': feedforward_block,
        'GlycN': glyc_n
    }

    with open('scripts/gsearch.yaml', 'w', encoding='utf8') as f:
        yaml.dump(grid, f)


def main():
    """Main function
    """

    grid = define_grid()

    # Iterate through all combinations of hyperparameters
    for params in product(*grid.values()):
        lr, epochs, batch_size, hidden_dim, kernel_size, out_channels = params

        # Write config file
        config = {
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size
            }
        write_config(config)

        # Run model and send stdout to log file
        logging.info('%s: Running model with params: %s, %s, %s, %s, %s, %s',
                     datetime.datetime.now(), *params)
        os.system('python scripts/kfold.py -c scripts/gsearch.yaml')


if __name__ == '__main__':
    main()
