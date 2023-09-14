"""Writes individual asparagine residues to a file.

__author__ = "Ben Iovino"
__date__ = "09/1/23"
"""

import os
import numpy as np
from Bio import SeqIO
from embed import Embedding, GlycEmb


def get_sites(sfile: str) -> dict:
    """Returns glycosylated asparagine residues for each sequence in a fasta file.

    :param sfile: fasta file
    :return dict: dictionary where key is seq ID and value is a dict of asparagine positions
    """

    seqs = {}
    for seq in SeqIO.parse(sfile, 'fasta'):
        glyc_pos = seq.description.split('\t')[1].split(':')  # Sites in fasta header
        glyc_tissue = seq.description.split('\t')[2]  # Tissue sources
        seqs[seq.id] = {}
        for i, res in enumerate(seq.seq):
            if res == 'N' and str(i+1) in glyc_pos:  # i+1 because indexing starts at 0
                seqs[seq.id][i] = 1  # 0 indexing as key to get embedding later on
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
        del n_seqs['sources']
        for pos, label in n_seqs.items():
            n_embed = GlycEmb(embed.id, embed.embed[pos], pos+1, label, sources)
            n_embeds.append(n_embed)

    # Write embeddings to file
    with open('data/nonglyc_embeds.npy', 'wb') as efile:  #pylint: disable=W1514
        np.save(efile, n_embeds)


def main():
    """Main
    """

    sfile = 'data/neg_seqs.txt'
    seqs = get_sites(sfile)
    print(seqs)
    edirec = 'data/neg_embeds'
    get_embeds(edirec, seqs)


if __name__ == '__main__':
    main()
