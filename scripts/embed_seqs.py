"""Embeds sequences using ESM2 model.

__author__ = "Ben Iovino"
__date__ = "08/28/23"
"""

import logging
import os
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


def embed_seqs(seqs: list):
    """Embeds a list of sequences and writes them to a file.

    :param seqs: list of sequences
    """

    model = Model()  # ESM2 encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Embed each sequence and write to file
    direc = 'data/embeds'
    if not os.path.exists(direc):
        os.makedirs(direc)
    for seq in seqs:

        # Skip existing embeddings
        if os.path.exists(f'{direc}/{seq[0]}.npy'):
            logging.info('Skipping %s', seq[0])
            continue

        # If sequence is too long, split into chunks
        if len(seq[1]) > 5000:
            logging.info('Splitting %s', seq[0])
            chunks = [seq[1][i:i+5000] for i in range(0, len(seq[1]), 5000)]

            # Initialize each split and embed
            embed = Embedding()
            for i, chunk in enumerate(chunks):
                embed.id, embed.seq = seq[0], chunk
                embed.esm2_embed(model, device, layer=17)
                embed.write(f'{direc}/{seq[0]}_{i}.npy')
            continue

        # Initialize object and embed
        logging.info('Embedding %s', seq[0])
        embed = Embedding(seq[0], seq[1])
        embed.esm2_embed(model, device, layer=17)
        embed.write(f'{direc}/{seq[0]}.npy')


def combine_embeds():
    """Combines split embeddings into one file.
    """

    # Get list of split embeddings
    embeds = {}
    prev_file = ''
    for file in sorted(os.listdir('data/embeds')):
        name = file.split('_')[0]
        if name == prev_file.split('_', maxsplit=1)[0]:
            embeds[name] = embeds.get(name, set()) | set([prev_file, file])
            embeds[name] = sorted(embeds[name])
        prev_file = file

    # Combine embeds for each key in dict
    for name, files in embeds.items():
        total_embed = Embedding()
        for file in files:
            part_embed = Embedding()
            part_embed.load(f'data/embeds/{file}')
            os.remove(f'data/embeds/{file}')
            total_embed.comb(part_embed)
        total_embed.write(f'data/embeds/{name}.npy')


def main():
    """Main
    """

    file = 'data/seqs.txt'
    seqs = load_seqs(file)
    embed_seqs(seqs)
    combine_embeds()


if __name__ == '__main__':
    main()
