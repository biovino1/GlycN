"""Parses file.

__author__ = "Ben Iovino"
__date__ = 11/17/2023
"""

import pandas as pd
import requests
from Bio import SeqIO


def parse_file(file: str) -> dict:
    """Returns a dictionary of proteins from file where each key is the accession ID and
    each value is a dict where each key is a glycosylated position and each value is a list
    of modifications and abundance values for that position.

    :param file: path to file
    :return dict: dictionary of dictionaries
    """

    serum_pdf = pd.read_excel(file, header=0)

    # Get unique protein ID's and all glycosylated positions
    prots = {}
    for i, prot in enumerate(serum_pdf['Master Protein Accessions']):

        # Get site and modification/abundance
        site = serum_pdf['Annotated Sequence'][i]
        mod = serum_pdf['Modifications'][i]
        abund = serum_pdf['Abundance'][i]

        # Add to dictionary
        prots[prot] = prots.get(prot, {})
        prots[prot][site] = prots[prot].get(site, [])
        prots[prot][site].append((mod, abund))

    return prots


def get_seqs(prots: dict):
    """Writes each accession ID and corresponding sequence to file.

    :param prots: dict of accession ID's and glycosites
    """

    # Request each fasta sequence from UniProt
    for seq in prots.keys():
        req = requests.get(f'https://www.uniprot.org/uniprot/{seq}.fasta', timeout=5)
        fasta = ''.join(req.text.split('\n')[1:])
        if fasta == '':
            continue

        # Write id, sites, and sequence to file
        with open('data/serum_seqs.fa', 'a', encoding='utf8') as sfile:
            sfile.write(f'>{seq}\n')
            sfile.write(f'{fasta}\n')


def site_pos(prots: dict) -> dict:
    """Returns a dictionary with substrings of sequence replaced with the position of the
    glycosylated amino acid.

    :param prots: dict of accession ID's and glycosites
    :return dict: updated dictionary
    """

    # Parse all seqs in serum.fa
    seqs = {}
    for record in SeqIO.parse('data/serum_seqs.fa', 'fasta'):
        seqs[record.id] = str(record.seq)

    # For each protein, replace each substring with the N's position in whole sequence
    for prot, substr in prots.items():
        positions = []
        for site in substr.keys():

            # Clean string and find its position in the whole sequence
            site = site.split('.')[1]
            n_pos = site.find('N')
            try:  # Some proteins not in sequence file if uniprot returns empty fasta seq
                positions.append(seqs[prot].find(site) + n_pos + 1)  # 1-indexed
            except KeyError:
                continue

        # Change names of keys
        keys = list(prots[prot].keys())
        for i, pos in enumerate(positions):
            key = keys[i]
            prots[prot][pos] = prots[prot].pop(key)

    return prots


def main():
    """Main function
    """

    file = 'data/Quantified_serum_glycopeptides_20230831.xlsx'
    prots = parse_file(file)
    get_seqs(prots)
    prots = site_pos(prots)
    print(prots['H0Y512'])


if __name__ == '__main__':
    main()
