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

        chars = set([c for c in self.corpus])
        chars = sorted(chars)
        self.char2index = {char: idx for idx, char in enumerate(chars)}
        self.index2char = {idx: char for idx, char in enumerate(chars)}
        self.chars_size = len(chars)

        self.input_corpus, self.label_corpus = self._corpus_set()

    def __len__(self):
        return len(self.input_corpus)

    def __getitem__(self, idx):
        source_indices = self.input_corpus[idx]
        target_index = self.label_corpus[idx]  # Note: It's a single index, not a list

        source = torch.LongTensor(source_indices)  # Use LongTensor for indices
        label = torch.LongTensor([target_index])  # Use LongTensor for a single index
        target = self._one_hot(label)

        return {
            'source': source,
            'label': label,
            'target': target
        }

    def _one_hot(self, label):
        one_hot = F.one_hot(label, num_classes=self.chars_size)
        one_hot = one_hot.squeeze(0)
        return one_hot.float()

    def _corpus_set(self):
        seq_size = self.sequence_length
        step = 1
        input_corpus, label_corpus = [], []
        for i in range(0, len(self.corpus) - seq_size, step):
            end_of_corpus = i + seq_size
            input_seq = [self.char2index[c] for c in self.corpus[i: end_of_corpus]]
            label = self.char2index[self.corpus[end_of_corpus]]  # Take the next character as a label
            input_corpus.append(input_seq)
            label_corpus.append(label)
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
        
        
if __name__ == '__main__':

    # Example usage:
    corpus_path = "/content/tinyGPT/corpus/Alice's_Adventures_in_Wonderland.txt"
    sequence_length = 10

    dataset = RnnDataset(corpus_path, sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        source = batch['source']
        label = batch['label']
        target = batch['target']
        print("Source Shape:", source.shape)
        print("Label Shape:", label.shape)
        print("Target Shape:", target.shape)
        break

