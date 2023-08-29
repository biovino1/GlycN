"""Parses the HumanAll.xlsx file and writes all of the unique fasta sequences to a file.

__author__ = "Ben Iovino"
__date__ = 08/28/2023
"""

import pandas as pd
import requests


def get_seqs(file: str):
    """Writes unique fasta sequences to a file along with accession ID and glycosites.

    :param file: Path to the file.
    :return: list of all the unique Accession ID's.
    """

    # Get all the unique accession ID's and unique glycosylation sites
    seqs = {}
    all_df = pd.read_excel(file, header=1)
    for i, accession in enumerate(all_df['Accession (UniProtKB)']):
        seqs[accession] = seqs.get(accession, set()) | set([all_df['Glycosylation location'][i]])

    # Request each fasta sequence from UniProt
    for seq, sites in seqs.items():
        sites = repr(sorted(sites)).strip('[]')  # Convert set to string
        req = requests.get(f'https://www.uniprot.org/uniprot/{seq}.fasta', timeout=5)
        fasta = ''.join(req.text.split('\n')[1:])

        # Write id, sites, and sequence to file
        with open('data/seqs.txt', 'a', encoding='utf8') as sfile:
            sfile.write(f'>{seq}    {sites}\n')
            sfile.write(f'{fasta}\n')


def main():
    """Main
    """

    file = 'data/HumanAll.xlsx'
    get_seqs(file)


if __name__ == '__main__':
    main()
