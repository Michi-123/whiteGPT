#@title word2vecライブラリーの読み込み
import io
import random
import re
import unicodedata
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# @title CBOWモデル
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.activation = torch.nn.Identity()

    def forward(self, context_indices):
        # コンテキスト単語の埋め込みベクトルの平均
        embeds = self.embeddings(context_indices).mean(dim=1)
        h = self.linear(embeds)
        h = self.activation(h)
        return h

#@title Vocab
class Vocab():
    def __init__(self, corpus):
        self.vocab = set(self.tokenize(corpus))
        self.word2index = {word: idx + 5 for idx, word in enumerate(self.vocab)}
        self.index2word = {idx + 5: token for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def add_special_tokens(self):
        BOS = '<BOS>'
        EOS = '<EOS>'
        PAD = '<PAD>'
        EXT1 = '<ext1>'
        EXT2 = '<ext2>'

        self.word2index[0] = BOS
        self.word2index[1] = EOS
        self.word2index[2] = PAD
        self.word2index[3] = EXT1 # 予備1
        self.word2index[4] = EXT2 # 予備2

        self.index2word[BOS] = 0
        self.index2word[EOS] = 1
        self.index2word[PAD] = 2
        self.index2word[EXT1] = 3 # 予備1
        self.index2word[EXT2] = 4 # 予備2

    def tokenize(self, corpus):
        corpus = corpus.lower()
        return re.findall(r'\w+|[^\w\s]', corpus)

#@title Custom Dataset
class TextDataset(Dataset):
    def __init__(self, vocab, corpus, window_size):
        self.corpus = corpus
        self.window_size = window_size
        self.vocab = vocab.vocab
        self.tokenize = vocab.tokenize
        self.word2index = vocab.word2index
        self.index2word = vocab.index2word
        self.tokenized_corpora = self._create_tokenized_corpora(corpus)

    def _create_tokenized_corpora(self, corpus):
        tokenized_corpora = []
        tokenized_corpus = self._create_tokenized_corpus(corpus)
        tokenized_line = []
        sequence_size = self.window_size + 1

        for i in range(len(tokenized_corpus) - sequence_size):
            tokenized_sequence = tokenized_corpus[i:i + sequence_size] #['は', '晴れ', 'です']
            tokenized_corpora.append(tokenized_sequence)

        return tokenized_corpora

    def _create_tokenized_corpus(self, corpus):
        corpus = corpus = self.tokenize(corpus)
        tokenized_corpus = [vocab.word2index[word] for word in corpus]
        return tokenized_corpus

    def tokenized_corpus2indices(self, tokenized_corpus):
        indices = []
        for word in tokenized_corpus:
            index = self.word2index[word]
            indices.append(index)
        return indices        

    def __len__(self):
        return len(self.tokenized_corpora)

    def __getitem__(self, idx):
        tokenized_corpus = self.tokenized_corpora[idx]
        source = tokenized_corpus[:self.window_size]
        target = tokenized_corpus[self.window_size]

        return {
            'source': torch.tensor(source),
            'target': torch.tensor(target),
        }

# 教材用にカスタマイズ 
class TextDataset(TextDataset):

    def test_corpus(self, test_corpus_list):
        lines = []

        for corpus in test_corpus_list:
            corpus = self.tokenize(corpus)
            line = [self.word2index[word] for word in corpus]
            lines.append(line) 
            
        self.tokenized_test_corpus = lines

    def test(self, model):
        # 行目の選択
        n = random.randint(0, len(self.tokenized_test_corpus)-1)
        line = self.tokenized_test_corpus[n]
        # 文章の選択
        m = random.randint(0,len(line)-1 - self.window_size)
        context = line[m : m + self.window_size]
        target = line[m + self.window_size]
        # インデックスに変換
        context_indices = [self.word2index[word] for word in context]
        context_indices = torch.LongTensor([context_indices])
        # 推論
        predicted_vector = model(context_indices)

        next_word_idx = torch.argmax(predicted_vector)
        next_word_idx = next_word_idx.squeeze().tolist()
        predicted_word = self.index2word[next_word_idx]
        print(' '.join(context),':',predicted_word,':', target)
        # join() メソッドは、配列の要素を指定された文字列で結合して、1つの文字列を返します。


"""
word2vec Functions
"""


""" コサイン類似度 """
import io
from tqdm import tqdm
import numpy as np

def load_vectors(fname):
    """
    学習済みモデルの読み込み
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}

    first_line = True

    for line in tqdm(fin):
        if first_line:
            first_line = False
            continue

        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=float)

    return data
