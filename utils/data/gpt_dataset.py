# -*- coding: utf-8 -*-
"""GptUtil.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fi2vSIEM7wVjSrghdecjsbwDMxv5OiEd
"""
import MeCab

import torch
from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence

tagger = MeCab.Tagger("-Owakati")

# @title TranslationDataset
class TranslationDataset(Dataset):
    def __init__(self, pairs, max_sequence_length):
        self.pairs = pairs
        self.max_sequence_length = max_sequence_length
        self.source_vocab, self.target_vocab = self.build_vocab()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_text, target_text = self.pairs[idx] # 'Hello how are you?',  'こんにち は 、 お 元気 です か ？'

        source_tokens = self.tokenize(source_text) # ['Hello', 'how', 'are', 'you?']
        target_tokens = self.tokenize(target_text)

        padded_source_tokens = self.pad_sequence(source_tokens, self.max_sequence_length) # ['Hello', 'how', 'are', 'you?', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
        padded_target_tokens = self.pad_sequence(target_tokens, self.max_sequence_length)

        source_indices = self.tokens_to_indices(padded_source_tokens, self.source_vocab)
        target_indices = self.tokens_to_indices(padded_target_tokens, self.target_vocab)

        return {
            'source_indices': torch.tensor(source_indices),
            'target_indices': torch.tensor(target_indices),
        }

    def build_vocab(self):
        source_texts, target_texts = zip(*self.pairs)

        source_tokens = [token for text in source_texts for token in self.tokenize(text)]
        target_tokens = [token for text in target_texts for token in self.tokenize(text)]

        source_unique_tokens = set(source_tokens + ['[PAD]'])  # [PAD] を追加
        target_unique_tokens = set(target_tokens + ['[PAD]'])  # [PAD] を追加

        source_vocab = {token: idx for idx, token in enumerate(source_unique_tokens)}
        target_vocab = {token: idx for idx, token in enumerate(target_unique_tokens)}

        return source_vocab, target_vocab

    def tokenize(self, text):
        # Simple tokenization, split by spaces
        return text.split()

    def pad_sequence(self, tokens, max_length):
        if len(tokens) < max_length:
            padding = ['[PAD]'] * (max_length - len(tokens))
            tokens += padding
        else:
            tokens = tokens[:max_length]
        return tokens

    def tokens_to_indices(self, tokens, vocab):
        return [vocab[token] for token in tokens]

#@title TextDataset
class TextDataset(Dataset):
    def __init__(self, corpus, max_sequence_length):
        self.corpus = corpus
        self.max_sequence_length = max_sequence_length
        self.source_vocab = self.build_vocab()
        vocab = set(word for sentence in corpus for word in sentence.split())
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        source_text = self.corpus[idx]
        source_tokens = self.tokenize(source_text)
        padded_source_tokens = self.pad_sequence(source_tokens, self.max_sequence_length)
        source_indices = self.tokens_to_indices(padded_source_tokens, self.source_vocab)
        return {
            'source_indices': torch.tensor(source_indices),
        }

    def build_vocab(self):
        source_texts = self.corpus
        source_tokens = [token for text in source_texts for token in self.tokenize(text)]
        source_unique_tokens = set(source_tokens + ['[PAD]'])  # [PAD] を追加
        source_vocab = {token: idx for idx, token in enumerate(source_unique_tokens)}
        return source_vocab

    def tokenize(self, text):
        # Simple tokenization, split by spaces
        return text.split()

    def pad_sequence(self, tokens, max_length):
        if len(tokens) < max_length:
            padding = ['[PAD]'] * (max_length - len(tokens))
            tokens += padding
        else:
            tokens = tokens[:max_length]
        return tokens

    def tokens_to_indices(self, tokens, vocab):
        return [vocab[token] for token in tokens]

import re
import unicodedata

#@title PrepareData
class PrepareData:

    def __init__(self, data_path='/content/data/eng-jpn.txt', max_length=15, use_filterPairs=True):

        self.tagger = tagger
        self.data_path = data_path
        self.max_length = max_length
        self.use_filterPairs = use_filterPairs

        self.reverse = True
        self.eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )

    def translation_data(self):
        print("読み込み中...")

        lines = open(self.data_path, encoding='utf-8').read().strip().split('\n')

        pairs = []
        for line in lines:

            pair = [] # 英語：日本語
            word = line.split('\t')
            pair.append(self._normalizeString(word[0]))

            # フランス・スペイン・ドイツ語対応
            # s2.append(self._normalizeString(s[1]))

            # 日本語対応
            watkati_gaki = self.tagger.parse(word[1]);
            watkati_gaki = watkati_gaki[:-1]
            pair.append(watkati_gaki)

            pairs.append(pair)

        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]

        # メモリー節約のために限定学習
        if self.use_filterPairs :pairs = self._filterPairs(pairs)

        print("%s個の文章のペアを読み込みます" % len(pairs))

        return pairs


    def gpt_data(self):
        pairs = self.translation_data()

        corpus = []
        for pair in pairs:
            corpus.append(pair[0])
            corpus.append(pair[1])

        return corpus

    def _normalizeString(self, s):
        s = self._unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def _unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def _filterPairs(self, pairs):
        return [pair for pair in pairs if self._filterPair(pair)]

    def _filterPair(self, p):
        return len(p[0].split(' ')) < self.max_length and \
            len(p[1].split(' ')) < self.max_length and \
            p[1].startswith(self.eng_prefixes)


#@title JpTextDataset
class JpTextDataset(Dataset):
    def __init__(self, corpus, max_sequence_length=15):

        self.tagger = tagger
        self.corpus = corpus
        self.max_sequence_length = max_sequence_length
        self.vocab = self.get_vocab(self.corpus)
        self.word2index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index2word = {idx:token for idx, token in enumerate(self.vocab)}
        self._add_padding_word()
        self.tokenized_corpus = self._tokenize_corpus(self.corpus)

    def _add_padding_word(self):
        self.vocab.append('[PAD]')
        index_pad = len(self.vocab) - 1
        self.word2index['[PAD]'] = index_pad
        self.index2word[index_pad] = ''

    def __len__(self):
        return len(self.tokenized_corpus)

    def __getitem__(self, idx):
        source_indices  = self.tokenized_corpus[idx]
        source_indices = torch.LongTensor(source_indices)
        return {
            'source': source_indices[:self.max_sequence_length],
            'target': source_indices[self.max_sequence_length]
        }

    def _tokenize_corpus(self, corpus):
        """
        全てのコーパスを指定の長さで区切って、行単位に格納します。
        """
        tokenized_corpus = []
        corpus = corpus.split() #['今日', 'は', '晴れ', 'です', '。']
        tokenized_line = []
        sequence_size = self.max_sequence_length + 1

        for i in range(len(corpus) - sequence_size):
            sequence = corpus[i:i + sequence_size] #['は', '晴れ', 'です']

            for token in sequence:
                index = self.word2index[token]
                tokenized_line.append(index)

                if len(tokenized_line) == sequence_size:
                    tokenized_corpus.append(tokenized_line)
                    tokenized_line = []

        return tokenized_corpus

    def get_vocab(self, corpus):
        tokens = set(token for token in corpus.split())
        tokens = sorted(tokens)
        return tokens

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

    def _read_jp_corpus(self, corpus_path):
        with open(corpus_path, 'r') as f:
            corpus = []
            for line in f:
                line = line.strip().lower()
                if len(line) == 0:
                    continue
                line = self.tagger.parse(line)
                corpus.append(line)
        corpus = " ".join(corpus)
        return corpus

    def sequence2indices(self, sequence):
        indices = []
        for word in sequence.split():
            index = self.word2index[word]
            indices.append(index)
        return indices

    def indices2sequence(self, indices):
        sequence = ''
        for index in indices:
            letter = self.index2word[index]
            sequence += letter
        return sequence


class JpSampleTextDataset(JpTextDataset):
    """
    青空文庫から芥川龍之介の作品をコーパスとしたデータセットに対応させるクラスです
    """
    def __init__(self):
        self.tagger = tagger
        self.max_sequence_length = 10
        self.word2index = self._set_obejct('word2index')
        self.index2word = self._set_obejct('index2word')
        self.pad = 6476

    def _set_obejct(self, object_path):
        import pickle
        import os
        base_path = '/content/whiteGPT/model/akutagawa/'
        
        with open(os.path.join(base_path, object_path),'rb') as f:
            object = pickle.load(f)
        return object

    def _create_attention_mask(self):
        s = self.max_sequence_length
        return (torch.triu(torch.ones((s, s)),1) == 0) * 1

    def sample(self, model, corpus, n=500, t=0.1):
        import copy
        import time
        from IPython.display import clear_output
        from IPython.core.display import display, HTML

        corpus = tagger.parse(corpus)
        source = self.sequence2indices(corpus)
        source = source[:self.max_sequence_length]
        indices = copy.copy(source)
        pad = self.word2index['[PAD]']
        mask = self._create_attention_mask()

        model.eval()
        display(HTML("<style>div.output_scroll { width: 100%; }</style>"))

        for _ in range(n):
            inputs = torch.LongTensor([source])
            outputs = model(inputs, mask)
            index = torch.argmax(outputs).item()
            indices.append(index)
            source.append(index)
            source = source[1:]
            text = self.indices2sequence(indices)
            display(HTML(text))
            clear_output(wait=True)
            
            time.sleep(t) # 視覚効果


#@title TranslationPreTrainDataset
class TranslationPreTrainDataset(Dataset):
    def __init__(self, corpus_path, max_sequence_length):
        self.tagger = tagger
        self.corpus = self._read_jp_corpus(corpus_path)
        self.max_sequence_length = max_sequence_length
        self.vocab = self.get_vocab(self.corpus)
        self.word2index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index2word = {idx:token for idx, token in enumerate(self.vocab)}
        self._add_padding_word()
        self.tokenized_corpus = self.tokenize_corpus(self.corpus)

    def _add_padding_word(self):
        self.vocab.append('[PAD]')
        index_pad = len(self.vocab) - 1
        self.word2index['[PAD]'] = index_pad
        self.index2word[index_pad] = ''

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        source_indices  = self.tokenized_corpus[idx]
        source_indices = torch.LongTensor(source_indices)
        return {
            'source': source_indices[:self.max_sequence_length],
            'target': source_indices[self.max_sequence_length]
        }

    def tokenize_corpus(self, corpus):
        """
        全てのコーパスを指定の長さで区切って、行単位に格納します。
        """
        tokenized_corpus = []
        corpus = corpus.split() #['今日', 'は', '晴れ', 'です', '。']
        tokenized_line = []
        sequence_size = self.max_sequence_length + 1

        for i in range(len(corpus) - sequence_size):
            sequence = corpus[i:i + sequence_size] #['は', '晴れ', 'です']

            for token in sequence:
                index = self.word2index[token]
                tokenized_line.append(index)

                if len(tokenized_line) == sequence_size:
                    tokenized_corpus.append(tokenized_line)
                    tokenized_line = []

        return tokenized_corpus

    def get_vocab(self, corpus):
        tokens = set(token for token in corpus.split())
        tokens = sorted(tokens)
        return tokens

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

    def _read_jp_corpus(self, corpus_path):
        corpus = ''
        with open(corpus_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                text1, text2 =  line.replace('\n', '').split('\t')
                corpus += text1 + text2
        return corpus

    def sequence2indices(self, sequence):
        indices = []
        for word in sequence.split():
            index = self.word2index[word]
            indices.append(index)
        return indices

    def indices2sequence(self, indices):
        sequence = ''
        for index in indices:
            letter = self.index2word[index]
            sequence += letter
        return sequence
