"""Parses file.

__author__ = "Ben Iovino"
__date__ = 11/17/2023
"""

import pandas as pd
import requests


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
        if prot in ['A0A087WTA8', 'A0A087X1N7']:  # These proteins don't have fa seq
            continue

        # Get site and modification/abundance
        site = serum_pdf['Annotated Sequence'][i]
        mod = serum_pdf['Modifications'][i]
        abund = serum_pdf['Abundance'][i]

        # Add to dictionary
        prots[prot] = prots.get(prot, {})
        prots[prot][site] = prots[prot].get(site, [])
        prots[prot][site].append((mod, abund))

    return prots


def get_seqs(prots: dict) -> dict:
    """Returns a dictionary of fasta sequences.

    :param prots: dict of accession ID's and glycosites
    :return dict: dictionary where key is accession ID and value is fasta sequence
    """

    # Request each fasta sequence from UniProt
    seqs = {}
    for seq in prots.keys():
        req = requests.get(f'https://www.uniprot.org/uniprot/{seq}.fasta', timeout=5)
        fasta = ''.join(req.text.split('\n')[1:])
        seqs[seq] = fasta

    return seqs


def site_pos(prots: dict, seqs: dict) -> dict:
    """Returns a dictionary with substrings of sequence replaced with the position of the
    glycosylated amino acid.

    :param prots: dict of accession ID's and glycosites
    :param seqs: dict of accession ID's and fasta sequences
    :return dict: updated dictionary
    """

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


def write_seqs(prots: dict, seqs: dict):
    """Writes sequences in a dictionary to one fasta file.

    :param prots: dict where key is accession ID and glycosites
    :param seqs: dict where key is accession ID and fasta sequence
    """

    # Write each sequence to file
    with open('data/test', 'w', encoding='utf8') as file:
        for prot, seq in seqs.items():

            # Get list of glycosites to write on fasta header line
            sites = list(prots[prot].keys())
            sites = ':'.join([str(s) for s in sites])
            file.write(f'>{prot}\t{sites}\n{seq}\n')


def main():
    """Main function
    """

    file = 'data/Quantified_serum_glycopeptides_20230831.xlsx'
    prots = parse_file(file)
    seqs = get_seqs(prots)
    prots = site_pos(prots, seqs)
    write_seqs(prots, seqs)


if __name__ == '__main__':
    main()
