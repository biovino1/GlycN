"""Parses the HumanAll.xlsx file and writes all of the unique fasta sequences to a file.

__author__ = "Ben Iovino"
__date__ = 08/28/2023
"""

import os
import urllib.request
import re
import pandas as pd
import requests


def qual_check(fasta: str, site: set) -> bool:
    """Returns True if fasta sequence is not empty and site corresponds to a valid
    glycosylation site. Returns False otherwise.

    :param fasta: fasta sequence
    :param site: set of glycosites
    :return bool: True if sequence is valid, False otherwise
    """

    # Check if fasta sequence is empty
    if fasta == '':
        return False

    # Check if glycosites are valid
    for s in site:
        sequon = fasta[s-1:s + 2]
        if not re.match(r'N[^P][ST]', sequon):
            return False

    return True


def get_seqs(sites: dict, sources: dict):
    """Writes each accession ID, glycosite, id resource, and fasta seq to file in
    fasta format.

    :param sites: dict of accession ID's and glycosites
    :param sources: dict of accession ID's and id resources
    """

    # Request each fasta sequence from UniProt
    for seq, site in sites.items():
        req = requests.get(f'https://www.uniprot.org/uniprot/{seq}.fasta', timeout=5)
        fasta = ''.join(req.text.split('\n')[1:])
        if not qual_check(fasta, site):
            continue

        # Set contains ints, so convert each one to string and join with colons
        site = [str(s) for s in sorted(site)]
        site = ':'.join(sorted(site))  # Converting sets to strings for writing
        source = ';'.join(sources[seq])

        # Write id, sites, and sequence to file
        with open('data/pos_seqs.fa', 'a', encoding='utf8') as sfile:
            sfile.write(f'>{seq}\t{site}\t{source}\n')
            sfile.write(f'{fasta}\n')


def get_info(file: str) -> tuple:
    """Returns two dicts, one containing all glycosites for each accession ID, and the other
    containing all identification resources.

    :param file: path to the file
    :return tuple: tuple of dicts, each where key is accession ID and value is a set
    """

    # Get all the unique accession ID's and unique glycosylation sites/identification resources
    sites, sources = {}, {}  # Easier to have two dicts for updating purposes
    all_df = pd.read_excel(file, header=1)
    for i, accession in enumerate(all_df['Accession (UniProtKB)']):

        # Update set of locations and resources for each accession
        location = set([all_df['Glycosylation location'][i]])  # one per line
        sites[accession] = sites.get(accession, set()) | location
        resources = set(all_df['Identification resources'][i].split(';'))  # mult per line
        sources[accession] = sources.get(accession, set()) | resources

    return sites, sources


def main():
    """Main
    """

    # Download file
    if not os.path.exists('data'):
        os.makedirs('data')
    url = 'http://nglycositeatlas.biomarkercenter.org/download/HumanAll/'
    file = 'data/HumanAll.xlsx'
    urllib.request.urlretrieve(url, file)

    # Get sequences
    sites, sources = get_info(file)
    get_seqs(sites, sources)


if __name__ == '__main__':
    main()
