# -*- coding: utf-8 -*-

# @title Transformer Dataset
import re
import unicodedata
import torch
from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    def __init__(self, tagger, data_path, context_size, prefixes):
        self.vocab_en = None
        self.vocab_ja = None
        self.src_data = None
        self.tgt_data = None

        self.BOS = 1
        self.EOS = 2

        self.tagger = tagger
        self.context_size = context_size
        self.prefixes = prefixes

        self._translation_dataset(data_path)

    def _translation_dataset(self, data_path):
        space = " "
        corpus_en = ""
        corpus_ja = ""
        sentences_en = []
        sentences_ja = []
        
        # 全データを行単位で読み込みます
        lines = self._read_lines(data_path)

        for line in lines:
            sentence_en, sentence_ja, _ = line.split('\t') # ３番目の要素は捨て要素
            # 正規化
            sentence_en = self._normalizeString(sentence_en)
            # 形態素解析
            sentence_ja = self.tagger.parse(sentence_ja)[:-1]
            # Filter処理
            if len(sentence_en.split()) > self.context_size:
                continue
            if len(sentence_ja.split()) > self.context_size:
                continue
            if not sentence_en.startswith(self.prefixes):
                continue

            # 行単位で文章を格納します
            sentences_en.append(sentence_en)
            sentences_ja.append(sentence_ja)
            
            # 文章を繋げます
            corpus_en += sentence_en + space
            corpus_ja += sentence_ja + space

        vocab_en = Vocab(corpus_en)
        vocab_ja = Vocab(corpus_ja)

        src_data = []
        tgt_data = []

        pairs = self._takenize(vocab_en, vocab_ja, sentences_en, sentences_ja)

        for pair in pairs:
            src_data.append(torch.LongTensor(pair[0]))
            tgt_data.append(torch.LongTensor(pair[1]))

        self.src_data = src_data
        self.tgt_data = tgt_data

        self.vocab_en = vocab_en
        self.vocab_ja = vocab_ja

    def _read_lines(self, data_path):
        lines = open(data_path, encoding='utf-8').read().strip().split('\n')
        return lines

    def _takenize(self, vocab_en, vocab_ja, sentences_en, sentences_ja):
        pairs = []
        for (sentence_en, sentence_ja) in zip(sentences_en, sentences_ja):
            indices_en = self.encode(vocab_en, sentence_en)
            indices_ja = self.encode(vocab_ja, sentence_ja, is_target=True)
            pairs.append([indices_en, indices_ja])
        return pairs

    def _normalizeString(self, sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", sentence)
        return sentence

    def encode(self, vocab, sentence,is_target=False):
        tokens = sentence.split()
        indices = [vocab.word2index[word] for word in tokens]
        if is_target:
            indices = indices + [self.EOS]
        indices = self._add_padding(indices)
        return indices

    def _add_padding(self, indices):
        indices = indices + [0] * self.context_size
        indices = indices[:self.context_size]
        return indices

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


# @title Vocab
class Vocab:
    def __init__(self, corpus):
        vocab =  sorted(set(corpus.split()))
        vocab.insert(0, '<PAD>')
        vocab.insert(1, '<BOS>')
        vocab.insert(2, '<EOS>')
        vocab.insert(3, '<UNK>')
        vocab.insert(4, '<EXT1>')
        vocab.insert(5, '<EXT2>')

        self.index2word = {idx: word for idx, word in enumerate(vocab)}
        self.word2index = {word: idx for idx, word in enumerate(vocab)}
        self.vocab_size = len(self.word2index)
