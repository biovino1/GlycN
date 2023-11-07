"""Defines the Embedding class, which is used to embed protein sequences using the
ESM-2_t36_3B protein language model.

__author__ = "Ben Iovino"
__date__ = "08/28/23"
"""

from dataclasses import dataclass
import esm
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


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
        self.embed = embed[0][1:-1]  # remove beginning and end tokens


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


@dataclass
class GlycDataset():
    """Prepares GlycEmb objects for training and testing.

    :param file: path to file with GlycEmbs
    :param train: percentage of data to use for training
    :param test: percentage of data to use for testing
    """
    file: str = ''
    data: np.ndarray = None


    def get_data(self):
        """Loads GlycEmb objects from a npy file and sets self.data to an equal number of
        positive and negative examples.
        """

        data = np.load(self.file, allow_pickle=True)

        # Separate positive and negative examples
        pos = [ex for ex in data if ex.label == 'pos']
        neg = [ex for ex in data if ex.label == 'neg']

        # Randomly undersample class with more examples
        if len(pos) > len(neg):
            pos = np.random.default_rng(1).choice(pos, len(neg), replace=False)
        else:
            neg = np.random.default_rng(1).choice(neg, len(pos), replace=False)

        self.data = np.concatenate((pos, neg), axis=0)


    def split(self, test: float):
        """Splits data into training and testing sets.
        
        :param test: percentage of data to use for testing
        """

        embeds = np.array([ex.emb for ex in self.data])
        labels = np.array([1 if ex.label == 'pos' else 0 for ex in self.data])

        # Reshape embeds so they are 1xn
        embeds = np.array([np.reshape(embed, (1, embed.shape[0])) for embed in embeds])

        # Split data
        embeds_train, embeds_test, labels_train, labels_test = train_test_split(
            embeds, labels, test_size=test, random_state=1)

        # Convert to tensors
        embeds_train = torch.from_numpy(embeds_train).float()
        embeds_test = torch.from_numpy(embeds_test).float()
        labels_train = torch.from_numpy(labels_train).long()
        labels_test = torch.from_numpy(labels_test).long()

        return embeds_train, embeds_test, labels_train, labels_test


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
