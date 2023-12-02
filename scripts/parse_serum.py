"""Parses the serum dataset to get each protein, it's sequence, each peptide with it's
glycosylated position, and the fucose concentration at that position.

__author__ = "Ben Iovino"
__date__ = 12/02/2023
"""

import pandas as pd
import requests


def parse_file(file: str) -> dict:
    """Returns a dict of proteins from file where each key is the accession ID and
    the value is a dict where each key is a glycosylated position within that protein
    and the value is it's fucose concentration

    :param file: path to file
    :return dict: dict of proteins
    """

    serum_pdf = pd.read_excel(file, header=0, sheet_name=3)

    # Get peptides for each protein and their respective fucose conc
    prots = {}
    for i, prot in enumerate(serum_pdf['Master Protein Accessions']):
        site = serum_pdf['Annotated Sequence'][i]  # peptide with glycosylated site
        fucose = serum_pdf['per fuc no mannose'][i]
        prots[prot] = prots.get(prot, {})
        prots[prot][site] = fucose

    return prots


def get_seqs(prots: dict) -> dict:
    """Returns a dictionary of fasta sequences.

    :param prots: dict of accession ID's and glycosites
    :return dict: dictionary where key is accession ID and value is fasta sequence
    """

    # Request each fasta sequence from UniProt
    seqs = {}
    for seq in prots.keys():
        print(seq)
        req = requests.get(f'https://www.uniprot.org/uniprot/{seq}.fasta', timeout=5)
        fasta = ''.join(req.text.split('\n')[1:])
        seqs[seq] = fasta

    return seqs


def site_pos(prots: dict, seqs: dict) -> dict:
    """Returns updated dictionary with each peptide's position in the protein.

    :param prots: dict of accession ID's and peptides
    :param seqs: dict of accession ID's and fasta sequences
    :return dict: updated dictionary
    """

    # For each peptide in each protein, find position of glycosylated amino acid
    for prot, peptides in prots.items():
        for pep in peptides.keys():

            # Clean string and find asparagine's position in entire sequence
            cl_pep = pep.split('.')[1]
            n_pos = cl_pep.find('N')
            pos = seqs[prot].find(cl_pep) + n_pos + 1  # 1-indexed

            # Add position to value of peptide
            prots[prot][pep] = (prots[prot][pep], pos)

    return prots


def write_seqs(prots: dict, seqs: dict):
    """Writes sequences in a dictionary to one fasta file.

    :param prots: dict where key is accession ID and peptides
    :param seqs: dict where key is accession ID and fasta sequence
    """

    # Write each sequence to file
    with open('data/serum_seqs.fa', 'w', encoding='utf8') as file:
        for prot, seq in seqs.items():

            # Get list of glycosites to write on fasta header line
            pos = ':'.join(set([str(prots[prot][pep][1]) for pep in prots[prot].keys()]))
            file.write(f'>{prot}\t{pos}\n{seq}\n')


def main():
    """Main function
    """

    file = 'data/Quantified_serum_glycopeptides_array_202309_07.xlsx'
    prots = parse_file(file)
    seqs = get_seqs(prots)
    prots = site_pos(prots, seqs)
    write_seqs(prots, seqs)


if __name__ == '__main__':
    main()
