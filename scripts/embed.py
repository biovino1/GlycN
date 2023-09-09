"""Defines the Embedding class, which is used to embed protein sequences using the
ESM-2_t36_3B protein language model.

__author__ = "Ben Iovino"
__date__ = "08/28/23"
"""

from dataclasses import dataclass
import esm
import numpy as np
import torch


class Model:
    """Stores model and tokenizer for embedding proteins.
    """

    def __init__(self):
        """Defines Model class.
        """

        self.encoder, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.tokenizer = self.alphabet.get_batch_converter()
        self.encoder.eval()  # disables dropout for deterministic results


    def to_device(self, device: str):
        """Moves model to device.

        :param device: cpu or gpu
        """

        self.encoder.to(device)


@dataclass
class Embedding:
    """Stores embeddings for a single protein sequence.

    :param id: sequence ID
    :param seq: protein sequence
    :param embed: embedding vector
    """
    id: str = ''
    seq: str = ''
    embed: np.ndarray = None


    def esm2_embed(self, model: Model, device: str, layer: int):
        """Returns embedding of a protein sequence. Each vector represents a single amino
        acid using Facebook's ESM2 model.

        :param seq: protein ID and sequence
        :param model: Model class with encoder and tokenizer
        :param device: gpu/cpu
        return: list of vectors
        """

        # Embed sequences
        self.seq = self.seq.upper()  # tok does not convert to uppercase
        embed = [np.array([self.id, self.seq], dtype=object)]  # for tokenizer
        _, _, batch_tokens = model.tokenizer(embed)
        batch_tokens = batch_tokens.to(device)  # send tokens to gpu

        with torch.no_grad():
            results = model.encoder(batch_tokens, repr_layers=[layer])
        embed = results["representations"][layer].cpu().numpy()
        self.embed = embed[0]


    def write(self, file: str):
        """Writes a single embedding to a file as an array. First index is id, second is the
        embedding.

        :param file: path to file
        """

        embed = np.array([self.id, self.seq, self.embed], dtype=object)
        with open(file, 'wb') as efile:  #pylint: disable=W1514
            np.save(efile, embed)


    def load(self, file: str):
        """Loads a single embedding from a file.

        :param file: path to file
        """

        with open(file, 'rb') as efile:
            embed = np.load(efile, allow_pickle=True)
        self.id = embed[0]
        self.seq = embed[1]
        self.embed = embed[2]


    def comb(self, emb):
        """Combines two embeddings.
        
        :param emb: Embedding class
        """

        # If self is empty, copy emb
        if self.id == '':
            self.id = emb.id
            self.seq = emb.seq
            self.embed = emb.embed
            return

        self.seq += emb.seq
        self.embed = np.concatenate(([self.embed, emb.embed]), axis=0)


@dataclass
class GlycEmb:
    """Stores a single embedding vector and several attributes.

    :param id: sequence ID
    :param emb: embedding vector
    :param pos: position of asparagine residue
    :param label: glycosylation label (1 = glycosylated, 0 = not glycosylated)
    :param sources: types of tissue
    """
    id: str = ''
    emb: np.ndarray = None
    pos: int = 0
    label: int = 0
    sources: str = ''
    cluster: int = 0
