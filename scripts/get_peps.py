"""Writes asparagine residues to a file with a window of surrounding amino acids.

__author__ = "Ben Iovino"
__date__ = "12/04/23"
"""

import argparse
from Bio import SeqIO


def get_unique(positions: list, concs: list) -> tuple:
    """Returns a list of unique positions.

    :param positions: list of positions
    :param concs: list of concentrations
    :return tuple: lists of unique positions and corresponding concentrations
    """

    # Get unique positions
    upos, uconcs = [], []
    for i, pos in enumerate(positions):
        if positions.count(pos) == 1:
            upos.append(int(pos))  # +1 for 0-indexing
            uconcs.append(concs[i])

    return upos, uconcs


def get_peps(file: str, win: int) -> dict:
    """Returns a dictionary of peptide sequences.

    :param file: fasta file
    :param win: window size around glycosylated residue
    :return dict: dictionary where key is accession ID and value is a list of peptides
    that contain glycosylated asparagine residues
    """

    # Get peptides for each sequence
    peps = {}
    for seq in SeqIO.parse(file, 'fasta'):

        # For each glycosylated positions, get peptide and fucose concentration
        positions = seq.description.split('\t')[1].split(':')
        concs = seq.description.split('\t')[2].split(':')
        positions, concs = get_unique(positions, concs)
        for i, pos in enumerate(positions):
            if seq.seq[pos-1] != 'N':  # Some positions are not N
                continue
            pep = str(seq.seq[pos-win-1:pos+win])
            if len(pep) != 2*win+1:  # Some peptides are too short
                continue
            peps[seq.id] = peps.get(seq.id, []) + [(pep, pos, concs[i])]

    return peps


def write_peps(peps: dict, file: str):
    """Writes a dictionary of peptide sequences to a fasta file.

    :param file: file to write peptides to
    :param peps: dictionary of peptides
    """

    with open(file, 'w', encoding='utf8') as pfile:
        for acc, peptides in peps.items():
            for i, pep in enumerate(peptides):
                pfile.write(f'>{acc}_{i}\t{pep[1]}\t{pep[2]}\n{pep[0]}\n')


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='data/serum_seqs.fa', help='sequence file')
    parser.add_argument('-p', type=str, default='data/serum_peps.fa', help='peptide file')
    parser.add_argument('-w', type=int, default=12, help='window size')
    args = parser.parse_args()

    # Get peptides and write each one to file
    peps = get_peps(args.f, args.w)
    write_peps(peps, args.p)


if __name__ == '__main__':
    main()
