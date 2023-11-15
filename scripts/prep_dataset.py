"""Prepares embeddings for training and testing.

__author__ = "Ben Iovino"
__date__ = 11/13/2023
"""

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def get_data(file: str, seed: int) -> np.ndarray:
    """Returns an array of GlycEmb objects with an equal amount of positive
    and negative examples.

    :param file: path to npy file
    :param seed: random seed
    """

    data = np.load(file, allow_pickle=True)

    # Separate positive and negative examples
    pos = [ex for ex in data if ex.label == 'pos']
    neg = [ex for ex in data if ex.label == 'neg']

    # Randomly undersample class with more examples
    if len(pos) > len(neg):
        pos = np.random.default_rng(seed).choice(pos, len(neg), replace=False)
    else:
        neg = np.random.default_rng(seed).choice(neg, len(pos), replace=False)
    data = np.concatenate((pos, neg), axis=0)

    return data


def split(data: np.ndarray, test: float) -> tuple:
    """Returns training and testing data.
    
    :param data: array of GlycEmb objects
    :param test: percentage of data to use for testing
    :return tuple: training and testing data
    """

    embeds = np.array([ex.emb for ex in data])
    labels = np.array([1 if ex.label == 'pos' else 0 for ex in data])

    # Split data
    embeds_train, embeds_test, labels_train, labels_test = train_test_split(
        embeds, labels, test_size=test, random_state=1)

    # Convert to tensors and write to file
    if not os.path.exists('data/datasets'):
        os.mkdir('data/datasets')
    torch.save(torch.from_numpy(embeds_train).float(), 'data/datasets/embeds_train.pt')
    torch.save(torch.from_numpy(embeds_test).float(), 'data/datasets/embeds_test.pt')
    torch.save(torch.from_numpy(labels_train).long(), 'data/datasets/labels_train.pt')
    torch.save(torch.from_numpy(labels_test).long(), 'data/datasets/labels_test.pt')


def main():
    """Main
    """

    data = get_data('data/N_embeds.npy', 1)
    split(data, 0.2)


if __name__ == '__main__':
    main()
