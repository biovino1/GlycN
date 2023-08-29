"""Embeds sequences using ESM2 model.

__author__ = "Ben Iovino"
__date__ = "08/28/23"
"""

import logging
import os
import numpy as np
import torch
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


def write_embeds(file: str, embed_list: list):
    """Writes multiple embeddings to a file as an array. For each entry, first index is id,
    second is the embedding.

    :param file: path to file
    :param embeds: list of embeddings
    """

    # For each index in list, replace with array of id and embedding
    for i, embed in enumerate(embed_list):
        embed_list[i] = np.array([embed.id, embed.embed], dtype=object)
    with open(file, 'wb') as efile:
        np.save(efile, embed_list)


def embed_seqs(seqs: list):
    """Embeds a list of sequences and writes them to a file.

    :param seqs: list of sequences
    """

    model = Model()  # ESM2 encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Embed each sequence and write to file
    embeds = []
    for seq in seqs:
        logging.info('Embedding %s', seq[0])
        embed = Embedding(seq[0], seq[1])
        embed.esm2_embed(model, device, layer=17)
        embeds.append(embed)
    write_embeds('data/embeds.npy', embeds)




def main():
    """Main
    """

    file = 'data/seqs.txt'
    seqs = load_seqs(file)
    embed_seqs(seqs)



if __name__ == '__main__':
    main()
