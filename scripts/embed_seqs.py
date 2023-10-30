"""Embeds sequences using ESM2 model.

__author__ = "Ben Iovino"
__date__ = "08/28/23"
"""

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
            if seq.seq == '':  # Skip seqs not found in UP
                continue
            seqs.append((seq.id, str(seq.seq)))

    return seqs


def split_embeds(seq: tuple, model: Model, device: str) -> Embedding:
    """Returns embedding of a protein sequence > 5000 residues.

    :param seq: tuple of ID and sequence
    :param model: Model class with encoder and tokenizer
    :param device: gpu/cpu
    """

    # Split sequence into chunks
    embeds = []
    chunks = [seq[1][i:i+3000] for i in range(0, len(seq[1]), 3000)]

    # Initialize each split and embed
    for chunk in chunks:
        embed = Embedding()
        embed.id, embed.seq = seq[0], chunk
        embed.esm2_embed(model, device, layer=17)
        embeds.append(embed)

    # Combine embeddings
    total_embed = Embedding()
    for embed in embeds:
        total_embed.comb(embed)

    return total_embed


def embed_seqs(seqs: list):
    """Embeds a list of sequences and writes them to a file.

    :param seqs: list of sequences
    """

    model = Model()  # ESM2 encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Embed each sequence and write to file
    embeds = []
    for i, seq in enumerate(seqs):

        # If sequence is too long, split into chunks
        if len(seq[1]) > 3000:
            logging.info('Splitting %s (%s)', seq[0], i)
            embed = split_embeds(seq, model, device)
            embeds.append(embed)
            continue

        # Initialize object and embed
        logging.info('Embedding %s (%s)', seq[0], i)
        embed = Embedding(seq[0], seq[1])
        embed.esm2_embed(model, device, layer=17)
        embeds.append(embed)

    with open('data/embeds.npy', 'wb') as dfile:
        np.save(dfile, embeds)


def main():
    """Main
    """

    file = 'data/all_seqs.txt'
    seqs = load_seqs(file)
    embed_seqs(seqs)


if __name__ == '__main__':
    main()
