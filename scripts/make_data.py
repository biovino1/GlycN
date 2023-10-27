"""Combines sequences from parse_all.py and parse_swissprot.py into a single file.

__author__ = "Ben Iovino"
__date__ = 08/28/2023
"""

import os
import parse_all
import parse_swissprot


def get_seqs(file: str) -> dict:
    """Returns a dict of sequences from both pos_seqs.txt and neg_seqs.txt

    :param file: path to the file
    :return: dict where key is ACC and value is a list of
      [position, source, glyc_label (pos/neg), sequence]
    """

    # Read and delete file
    with open(file, 'r', encoding='utf8') as sfile:
        seqs = sfile.readlines()
    os.remove(file)

    # Make dict
    seq_dict = {}
    label = file.split('_')[0].split('/')[-1]
    for i, seq in enumerate(seqs):
        if seq[0] != '>':
            continue
        seq = seq.split('\t')
        seq_dict[seq[0].strip('>')] = [seq[1], seq[2].strip(), label, seqs[i+1].strip()]

    return seq_dict


def main():
    """
    """

    # Call parse_all.py and parse_swissprot.py
    parse_all.main()
    parse_swissprot.main()

    # Read sequences from both files and combine dictionaries
    pos_seqs = get_seqs('data/pos_seqs.txt')
    neg_seqs = get_seqs('data/neg_seqs.txt')
    seqs = {**pos_seqs, **neg_seqs}

    # Write to file
    with open('data/all_seqs.txt', 'w', encoding='utf8') as afile:
        for acc, info in seqs.items():
            afile.write(f'>{acc}\t{info[0]}\t{info[1]}\t{info[2]}\n{info[3]}\n')


if __name__ == '__main__':
    main()
