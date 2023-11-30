"""Embeds sequences using ESM2 model.

__author__ = "Ben Iovino"
__date__ = "08/28/23"
"""

import argparse
import datetime
import logging
import os
import torch
import numpy as np
from Bio import SeqIO
from embed import Model, Embedding


log_filename = 'data/logs/embed_seqs.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def load_seqs(file: str) -> list:
    """Returns a list of sequences and their IDs froma fasta file.

    :param file: fasta file
    :return list: list of sequences
    """

    # Read each line and add to list
    seqs = []
    with open(file, 'r', encoding='utf8') as f:
        for seq in SeqIO.parse(f, 'fasta'):
            seqs.append((seq.id, str(seq.seq)))

    return seqs


def embed_seqs(seqs: list, efile: str):
    """Embeds a list of sequences and writes them to a file.

    :param seqs: list of sequences
    :param efile: path to embeddings file to save
    """

    model = Model()  # ESM2 encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Embed each sequence and write to file
    embeds = []
    for i, seq in enumerate(seqs):

        # If device is on CPU, move to GPU
        if device == torch.device('cpu'):
            device = torch.device('cuda')
            model.to_device(device)

        # If sequence is too long, move device to CPU
        if len(seq[1]) > 3000:
            device = torch.device('cpu')
            model.to_device(device)
            logging.info('Sequence %s too long, moving to CPU', seq[0])

        # Initialize object and embed
        logging.info('%s: Embedding %s (%s)', datetime.datetime.now(),seq[0], i)
        embed = Embedding(seq[0], seq[1])
        embed.esm2_embed(model, device, layer=17)
        embeds.append(embed)

    with open(efile, 'wb') as file:
        np.save(file, embeds)


def main():
    """Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='data/all_seqs.fa', help='fasta file')
    parser.add_argument('-e', type=str, default='data/embeds.npy', help='embeddings file')
    args = parser.parse_args()

    # Load sequences from file and embed
    seqs = load_seqs(args.f)
    embed_seqs(seqs, args.e)


if __name__ == '__main__':
    main()
