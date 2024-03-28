#@title word2vecライブラリーの読み込み
import io
import random
import torch
import torch.nn as nn
import tqdm

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


"""
word2vec Functions
"""

window_size = None
vocab = None
vocab_size = None
word2index = None
index2word = None
tokenized_corpus = None
tokenized_test_corpus = None

# テキストデータの前処理
def preprocess_text(text):
    return text.lower().split()


def context2indices(context):
    return [word2index[word] for word in context]

def set_tokenized_test_corpus(test_corpus):
    global tokenized_test_corpus
    tokenized_test_corpus = [preprocess_text(sentence) for sentence in test_corpus]

def test(model):
    # 行の選択
    n = random.randint(0,len(tokenized_test_corpus)-1)
    line = tokenized_test_corpus[n]
    # 文章の選択
    m = random.randint(0,len(line)-1 - window_size)
    context = line[m:m+window_size]
    target = line[m+window_size]
    # インデックスに変換
    context_indices = [word2index[word] for word in context]

    context_indices = torch.LongTensor([context_indices])
    # 推論
    predicted_vector = model(context_indices)

    next_word_idx = torch.argmax(predicted_vector)
    next_word_idx = next_word_idx.squeeze().tolist()
    predicted_word = index2word[next_word_idx]
    print(' '.join(context),':',predicted_word,':', target)
    # join() メソッドは、配列の要素を指定された文字列で結合して、1つの文字列を返します。



# 入力と教師データを作る関数です
def dataset(idx, sentence):

    context_start = idx # 始点のインデックス
    context_end = idx + window_size # 終点のインデックス
    context = [sentence[i] for i in range(context_start, context_end)]
    target_word = sentence[idx + window_size]

    context_indices = context2indices(context)
    target_index = word2index[target_word]

    context_indices = torch.LongTensor([context_indices])
    target_index = torch.LongTensor([target_index])
    return context_indices, target_index

def tokenize(corpus):
    global vocab, vocab_size, word2index, index2word, tokenized_corpus
    # コーパスの前処理
    tokenized_corpus = [preprocess_text(sentence) for sentence in corpus]
    # ボキャブラリの構築
    vocab = set(word for sentence in tokenized_corpus for word in sentence)

    # ワードのインデックス付け
    word2index = {word: idx for idx, word in enumerate(vocab)}
    index2word = {idx: word for word, idx in word2index.items()}
    vocab_size = len(vocab)
    vocab_size

    return tokenized_corpus


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
