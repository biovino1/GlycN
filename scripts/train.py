"""Trains model defined on model.py using data from N_embeds.npy


__author__ = "Ben Iovino"
__date__ = "10/31/23"
"""

from embed import GlycDataset
from model import GlycN
import yaml


def main():
    """Main
    """

    # Load data
    dataset = GlycDataset('data/N_embeds.npy')
    dataset.get_data()
    embeds_train, embeds_test, labels_train, labels_test = dataset.split(0.2)

    # Define model parameters
    with open('scripts/config.yaml', 'r', encoding='utf8') as cfile:
        config = yaml.safe_load(cfile)

    # Train model
    model = GlycN(config)


if __name__ == '__main__':
    main()
