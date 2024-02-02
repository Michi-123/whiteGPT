# -*- coding: utf-8 -*-
"""RnnUtil.ipynb

"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#@title RnnDataset
class RnnDataset(Dataset):
    def __init__(self, corpus_path, sequence_length):
        self.corpus = self._read_corpus(corpus_path)
        self.sequence_length = sequence_length

        chars = set([c for c in self.corpus]) # 一覧
        chars = sorted(chars)
        self.char2index = {char: idx for idx, char in enumerate(chars)}
        self.index2char = {index: char for index, char in enumerate(chars)}
        self.chars_size = len(chars)
        
        input_corpus, label_corpus = self._corpus_set()
        self.input_corpus = input_corpus
        self.label_corpus = label_corpus

    def __len__(self):
        return len(self.input_corpus)

    def __getitem__(self, idx):
        source_indices = self._corpus_to_indices(self.input_corpus[idx], self.char2index)
        target_indices = self._corpus_to_indices(self.label_corpus[idx], self.char2index)

        source = torch.FloatTensor(source_indices)
        label = torch.FloatTensor(target_indices)
        target = self._one_hot(label)

        return {
            'source': source,
            'label': label,
            'target': target
        }

    def _one_hot(self, label):
        one_hot = F.one_hot(label.long(), num_classes=self.chars_size)
        one_hot = one_hot.squeeze(0)
        return one_hot.float()

    def _corpus_set(self):
        seq_size = self.sequence_length
        step = 1
        input_corpus, label_corpus = [], []
        # Convert the data into a series of different SEQLEN-length subsequences.
        for i in range(0, len(self.corpus) - seq_size, step):
            end_of_corpus = i + seq_size
            input_corpus.append(self.corpus[i: end_of_corpus])
            label_corpus.append(self.corpus[end_of_corpus]) #次の一文字
        return input_corpus, label_corpus

    def _read_corpus(self, corpus_path):
        with open(corpus_path, 'rb') as f:
            lines = []
            for line in f:
                line = line.strip().lower().decode("ascii", "ignore")
                if len(line) == 0:
                    continue
                lines.append(line)
        corpus = " ".join(lines)
        return corpus

    def _corpus_to_indices(self, corpus, char2index):
        return [char2index[letter] for letter in corpus]

    def indices2sequence(self, indices):
        sequence = ''
        for index in indices:
            letter = self.index2char[index]
            sequence += letter
        return sequence

    def sequence2indices(self, sequence):
        sequence = sequence[:self.sequence_length]
        sequence = sequence.lower()
        indices = []
        for char in sequence:
            index = self.char2index[char]
            indices.append(index)
        return indices