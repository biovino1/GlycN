"""Parses the Swissprot_Train_Validation_dataset.csv file and writes all of the
nuclei and mitrochondria sequences to a file.

__author__ = "Ben Iovino"
__date__ = 08/28/2023
"""

import csv
import os
import re
import urllib.request


def check_locs(row: list) -> tuple:
    """Checks a row from csv file to see if it is a nuclear or mitochondrial protein

    :param row: row from csv in a list
    :return: tuple of (mito, nuclear) where mito and nuclear are booleans
    """

    mito, nuclear = False, False
    for i, index in enumerate(row[4:15]):
        if index == '1.0':
            if i not in [2, 5]:  # If in any other index, doesn't count
                mito, nuclear = False, False
                break
            if i == 2:  # mt index
                mito = True
            if i == 5:  # nucleus index
                nuclear = True

    return mito, nuclear


def get_seqs(file: str) -> dict:
    """Returns a dict of sequences that are either nuclear or mitochondrial.

    :param file: path to the file
    :return: dict where key is ACC and value is tuple of [mito, nuclear, sequence]
    """

    # Read csv file
    seqs = {}
    with open(file, 'r', encoding='utf8') as cfile:
        reader = csv.reader(cfile)
        next(reader)  # Skip header
        for row in reader:

            # Get sequence if nuclear or mitochondrial
            mito, nuclear = check_locs(row)
            if mito or nuclear:
                seqs[row[1]] = [mito, nuclear, row[15]]

    return seqs


def get_sequons(seqs: dict) -> dict:
    """Returns dict of sequences with starting index of sequons appended.

    :param seqs: dict where key is ACC and value is list of [mito, nuclear, sequence]
    :return: dict same as param but with starting index of sequons appended
    """

    del_list = []
    for acc, tup in seqs.items():
        seq = tup[2]
        matches = re.finditer(r'N[^P][ST]', seq)
        for match in matches:
            seqs[acc].append(match.start()+1)  # +1 because indexing starts at 0

        # If no sequons were added to the list, delete the accession
        if len(seqs[acc]) == 3:
            del_list.append(acc)

    # Can't delete while iterating so delete here
    for acc in del_list:
        del seqs[acc]

    return seqs


def write_seqs(seqs: dict):
    """Writes sequences in a dictionary to one fasta file.

    :param seqs: dict where key is ACC and value is list of [mito, nuclear, sequence]
    """

    for seq, values in seqs.items():

        # Get subcellular locations
        sources = []
        if values[0]:  # if mt is True
            sources.append('mitochondria')
        if values[1]:  # if nucleus is True
            sources.append('nucleus')
        sources = ';'.join(sources)

        # Get sequon locations
        sites = values[3:]
        sites = ':'.join([str(s) for s in sites])  # Convert ints to strings for writing

        # Write id, subcellular location, sequon locations, and sequnce to file
        with open('data/neg_seqs.fa', 'a', encoding='utf8') as sfile:
            sfile.write(f'>{seq}\t{sites}\t{sources}\n')
            sfile.write(f'{values[2]}\n')


def main():
    """Main
    """

    # Download file
    if not os.path.exists('data'):
        os.makedirs('data')
    url = 'https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/' \
        'Swissprot_Train_Validation_dataset.csv'
    file = 'data/Swissprot_Train_Validation_dataset.csv'
    urllib.request.urlretrieve(url, file)

    # Get sequences
    seqs = get_seqs(file)
    seqs = get_sequons(seqs)
    write_seqs(seqs)


if __name__ == '__main__':
    main()
