"""Defines the Embedding class, which is used to embed protein sequences using the
ESM-2_t36_3B protein language model.

__author__ = "Ben Iovino"
__date__ = "08/28/23"
"""

from dataclasses import dataclass
import esm
import numpy as np
import regex as re
import torch
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import Dataset


class Model:
    """Stores model and tokenizer for embedding proteins.
    """

    def __init__(self, model: str):
        """Model contains encoder and tokenizer.
        """

        if model == 'esm2':
            self.load_esm2()
        elif model == 'prott5':
            self.load_prott5xl()


    def load_esm2(self):
        """Loads ESM-2 model.
        """

        self.encoder, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.tokenizer = alphabet.get_batch_converter()
        self.encoder.eval()


    def load_prott5xl(self):
        """Loads ProtT5-XL model.
        """

        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50',
                                                    do_lower_case=False)
        self.encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")


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
        """

        # Embed sequences
        self.seq = self.seq.upper()  # tok does not convert to uppercase
        embed = [np.array([self.id, self.seq], dtype=object)]  # for tokenizer
        _, _, batch_tokens = model.tokenizer(embed)
        batch_tokens = batch_tokens.to(device)  # send tokens to gpu

        with torch.no_grad():
            results = model.encoder(batch_tokens, repr_layers=[layer])
        embed = results["representations"][layer].cpu().numpy()
        self.embed = embed[0][1:-1]  # remove beginning and end tokens


    def prott5xl_embed(self, model: Model, device: str):
        """Returns embedding of a protein sequence. Each vector represents a single amino
        acid using Rostlab's ProtT5-XL model.
        """

        # Remove special chars, add space after each amino acid so each residue is vectorized
        seq = re.sub(r"[UZOB]", "X", self.seq)
        seq = [' '.join([*seq])]

        # Tokenize, encode, and load sequence
        ids = model.tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)  # pylint: disable=E1101
        attention_mask = torch.tensor(ids['attention_mask']).to(device)  # pylint: disable=E1101

        # Extract sequence features
        with torch.no_grad():
            embedding = model.encoder(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        # Remove padding and special tokens
        features = []
        for seq_num in range(len(embedding)):  # pylint: disable=C0200
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            features.append(seq_emd)
        self.embed = features[0]


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


@dataclass
class GlycEmb:
    """Stores a single embedding vector and several attributes.

    :param id: sequence ID
    :param emb: embedding vector for asparagine residue
    :param pos: position of asparagine residue in protein sequence (1-indexed)
    :param label: glycosylation label (pos for glycosylated, neg for non-glycosylated)
    :param sources: subcellular location or tissue type
    """
    id: str = ''
    emb: np.ndarray = None
    pos: int = 0
    label: str = ''
    sources: str = ''


class PytorchDataset(Dataset):
    """Custom dataset for training and testing Pytorch models.
    """

    def __init__(self, embeds, labels):
        """Defines CustomDataset class.

        :param X: array of embeddings
        :param y: array of labels
        """

        self.embeds = embeds
        self.labels = labels


    def __len__(self):
        """Returns length of dataset.
        """

        return len(self.embeds)


    def __getitem__(self, idx):
        """Returns embed and label at index idx.
        """

        sample = {'embed': self.embeds[idx],
            'label': self.labels[idx]}

        return sample
