"""Parses file.

__author__ = "Ben Iovino"
__date__ = 11/17/2023
"""

import pandas as pd
import requests


def parse_file(file: str):
    """

    :param file: path to file
    """

    serum_pdf = pd.read_excel(file, header=0)

    # Get unique protein ID's and all glycosylated positions
    prots = {}
    for i, prot in enumerate(serum_pdf['Master Protein Accessions']):
        site = dict(serum_pdf['Position in Protein'][i])
        prots[prot] = prots.get(prot, set()) | site


def main():
    """Main function
    """

    file = 'data/Quantified_serum_glycopeptides_20230831.xlsx'
    parse_file(file)


if __name__ == '__main__':
    main()
