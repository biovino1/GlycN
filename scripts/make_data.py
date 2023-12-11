"""Combines sequences from parse_all.py and parse_swissprot.py into a single file.

__author__ = "Ben Iovino"
__date__ = 08/28/2023
"""

import parse_all
import parse_swissprot
from Bio import SeqIO


def get_seqs(file: str) -> dict:
    """Returns a dict of sequences from a fasta file.

    :param file: path to the file
    :return: dict where key is ACC and value is a list of
      [position, source, glyc_label (pos/neg), sequence]
    """

    seqs = {}
    label = file.split('_')[0].split('/')[-1]  # pos or neg
    for seq in SeqIO.parse(file, 'fasta'):
        desc = seq.description.split('\t')
        seqs[seq.id] = [desc[1], desc[2], label, str(seq.seq)]

    return seqs


def main():
    """Main function
    """

    # Call parse_all.py and parse_swissprot.py
    parse_all.main()
    parse_swissprot.main()

    # Read sequences from both files and combine dictionaries
    pos_seqs = get_seqs('data/pos_seqs.fa')
    neg_seqs = get_seqs('data/neg_seqs.fa')
    seqs = {**pos_seqs, **neg_seqs}

    # Write to file
    with open('data/all_seqs.fa', 'w', encoding='utf8') as afile:
        for acc, info in seqs.items():
            afile.write(f'>{acc}\t{info[0]}\t{info[1]}\t{info[2]}\n')
            afile.write(f'{info[3]}\n')


if __name__ == '__main__':
    main()
