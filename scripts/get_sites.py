"""Writes individual asparagine residues to a file.

__author__ = "Ben Iovino"
__date__ = "09/1/23"
"""

import os
import numpy as np
from Bio import SeqIO
from embed import Embedding, GlycEmb


def get_sites(sfile: str) -> dict:
    """Returns sequon positions and sources for each sequence in a fasta file.

    :param sfile: fasta file
    :return dict: dictionary where key is seq ID and value is a dict with sequon positions
    and subcellular/tissue sources
    """

    seqs = {}
    for seq in SeqIO.parse(sfile, 'fasta'):
        seqs[seq.id] = {}
        glyc_pos = seq.description.split('\t')[1].split(':')  # Sites in fasta header
        seqs[seq.id]['glyc_pos'] = [int(pos) for pos in glyc_pos]
        glyc_tissue = seq.description.split('\t')[2]  # Tissue sources
        seqs[seq.id]['sources'] = glyc_tissue

    return seqs


def get_embeds(edirec: str, seqs: dict):
    """Writes embeddings for each asparagine residue in each sequence to one file.
    
    :param edirec: directory containing embeddings
    :param seqs: dictionary of asparagine positions, glycosylation labels, and tissue sources
    """

    n_embeds = []
    for efile in os.listdir(edirec):
        embed = Embedding()
        embed.load(os.path.join(edirec, efile))

        # Get embeddings for each asparagine residue
        n_seqs = seqs[embed.id]
        sources = n_seqs['sources']
        for pos in seqs[embed.id]['glyc_pos']:
            n_embed = GlycEmb(embed.id, embed.embed[pos], pos, sources, 1)
            n_embeds.append(n_embed)

    # Write embeddings to file
    with open('data/nonglyc_embeds.npy', 'wb') as efile:  #pylint: disable=W1514
        np.save(efile, n_embeds)


def main():
    """Main
    """

    sfile = 'data/neg_seqs.txt'
    seqs = get_sites(sfile)
    edirec = 'data/neg_embeds'
    get_embeds(edirec, seqs)


if __name__ == '__main__':
    main()
