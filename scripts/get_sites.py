"""Writes individual asparagine residues to a file.

__author__ = "Ben Iovino"
__date__ = "09/1/23"
"""

import argparse
import numpy as np
from Bio import SeqIO
from embed import GlycEmb


def get_sites(sfile: str) -> dict:
    """Returns sequon positions and sources for each sequence in a fasta file.

    :param sfile: fasta file
    :return dict: dictionary where key is seq ID and value is a dict with sequon positions
    and subcellular/tissue sources
    """

    seqs = {}
    for seq in SeqIO.parse(sfile, 'fasta'):
        seqs[seq.id] = {}
        info = seq.description.split('\t')
        glyc_pos = info[1].split(':')  # Sites in fasta header
        seqs[seq.id]['glyc_pos'] = [int(pos) for pos in glyc_pos]
        glyc_tissue = info[2]  # Tissue sources
        seqs[seq.id]['sources'] = glyc_tissue
        seqs[seq.id]['label'] = info[3]

    return seqs


def get_embeds(efile: str, nfile: str, seqs: dict):
    """Writes embeddings for each asparagine residue in each sequence to one file.
    
    :param efile: file containing embeddings
    :param nfile: file containing individual asparagine embeddings
    :param seqs: dictionary of asparagine positions, glycosylation labels, and tissue sources
    """

    # Load embeddings
    embeddings = np.load(efile, allow_pickle=True)

    # Get individual embeddings for each asparagine residue
    n_embeds = []
    for embed in embeddings:

        # Get embeddings for each asparagine residue
        n_seqs = seqs[embed.id]
        sources = n_seqs['sources']
        for pos in seqs[embed.id]['glyc_pos']:
            n_embed = GlycEmb(embed.id, embed.embed[pos-1], pos, n_seqs['label'], sources)
            n_embeds.append(n_embed)

    # Write embeddings to file
    with open(nfile, 'wb') as file:  #pylint: disable=W1514
        np.save(file, n_embeds)


def main():
    """Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='data/all_seqs.fa', help='fasta file')
    parser.add_argument('-e', type=str, default='data/embeds.npy', help='embeddings file')
    parser.add_argument('-n', type=str, default='data/N_embeds.npy', help='N embeddings file')
    args = parser.parse_args()

    # Parse fasta file for sites and save individual Asp embeddings
    seqs = get_sites(args.f)
    get_embeds(args.e, args.n, seqs)


if __name__ == '__main__':
    main()
