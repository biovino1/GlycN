"""Prepares embeddings for training and testing.

__author__ = "Ben Iovino"
__date__ = 11/13/2023
"""

import argparse
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from Bio import SeqIO


def cluster_seqs(embs: list, label: str, ct: float) -> list:
    """Returns sequences clustered by MMseqs2.

    :param embs: list of GlycEmb objects
    :param label: pos or neg
    :param ct: sequence identity threshold for clustering
    :return list: GlycEmb's that are found in clustered sequences
    """

    # Cluster corresponding sequences
    os.system(f'mmseqs easy-cluster data/{label}_seqs.fa data/cluster_seqs tmp --min-seq-id {ct}')
    reps = {}
    for seq in SeqIO.parse('data/cluster_seqs_rep_seq.fasta', 'fasta'):
        reps[seq.id] = str(seq.seq)
    os.system('rm -rf tmp data/cluster_seqs*')

    # Return GlycEmb's that are in reps
    clustered = [ex for ex in embs if ex.id in reps]

    return clustered


def get_data(cluster: bool, file: str, seed: int, ct: float) -> np.ndarray:
    """Returns an array of GlycEmb objects with an equal amount of positive
    and negative examples.

    :param cluster: whether to cluster dataset
    :param file: path to npy file
    :param seed: random seed
    :param ct: clustering threshold
    :return np.ndarray: array of GlycEmb objects
    """

    data = np.load(file, allow_pickle=True)

    # Separate positive and negative examples
    pos = [ex for ex in data if ex.label == 'pos']
    neg = [ex for ex in data if ex.label == 'neg']

    # Cluster dataset
    if cluster:
        pos = cluster_seqs(pos, 'pos', ct)
        neg = cluster_seqs(neg, 'neg', ct)

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
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=bool, default=True, help='cluster dataset')
    parser.add_argument('-f', type=str, default='data/esm2_17_N_embeds.npy')
    parser.add_argument('-s', type=int, default=1, help='random seed')
    parser.add_argument('-t', type=float, default=0.3, help='clustering threshold')
    args = parser.parse_args()

    data = get_data(args.c, args.f, args.s, args.t)
    split(data, 0.2)


if __name__ == '__main__':
    main()
