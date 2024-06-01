# -*- coding: utf-8 -*-
"""GptUtil.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fi2vSIEM7wVjSrghdecjsbwDMxv5OiEd
"""
# @title GptDataset

import re
import unicodedata
import torch
from torch.utils.data import Dataset, DataLoader

tagger = None


# @title Vocab
class Vocab:
    def __init__(self, corpus):
        #vocab =  sorted(set(' '.join(sentences).split()))
        vocab =  sorted(set(corpus.split()))
        vocab.insert(0, '<PAD>')
        vocab.insert(1, '<BOS>')
        vocab.insert(2, '<EOS>')
        vocab.insert(3, '<UNK>')
        vocab.insert(4, '<EXT1>')
        vocab.insert(5, '<EXT2>')

        # self.max_size = 5000 必要であれば追加
        self.index2word = {idx: word for idx, word in enumerate(vocab)}
        self.word2index = {word: idx for idx, word in enumerate(vocab)}
        self.vocab_size = len(self.word2index)
        self.word_freq = {}
        self.word_freq_desc = {}

        # 頻出度を更新
        #self._update_word_freq(corpus)
        self._create_word_freq(corpus)

    def add_vocab(self, corpus):
        added_vocab = sorted(set(corpus.split()))
        vocab_size = self.vocab_size

        index2word = {idx + vocab_size: word for idx, word in enumerate(added_vocab)}
        word2index = {word: idx + vocab_size for idx, word in enumerate(added_vocab)}

        self.index2word.update(index2word)
        self.word2index.update(word2index)

        self.vocab_size = len(self.word2index)

        self._update_word_freq(corpus)

    def _update_word_freq(self, corpus):
        word_freq = self.word_freq

        """ 新しいコーパスから単語の頻度を更新 """
        for word in corpus.split():
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1

        # 頻出度でソート
        self.word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=False))
        self.word_freq_desc = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

    def __truncate_vocab_CUDA_error(self, minimum_frequency_count=1):
        """ 実行するとCUDA kernel errors. """

        for index in range(6, len(self.word2index)):
            word = self.index2word[index]
            frequency_count = self.word_freq_desc[word]
            if frequency_count == minimum_frequency_count:
                del self.index2word[index]
                del self.word2index[word]
                # 以下でも同様のエラー
                # index2word.pop(1, 'Key not found')
                # word2index.pop(word, 'Key not found')
                self.vocab_size -= 1

    def _create_word_freq(self, corpus):

        word_freq = {}

        """ 新しいコーパスから単語の頻度を更新 """
        for word in corpus.split():
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1

        # 頻出度でソート
        self.word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=False))
        self.word_freq_desc = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

    def remove_rare_words(self, degree=1):
        #self._make_freq(corpus)
        self._remove_rare_words(degree)
        self._reconstruct_vocab()
        print('新しいインデックスを作成しました')

    def _remove_rare_words(self, degree=1):
        word_freq = self.word_freq
        # 頻出度が1の要素を削除
        for word, freq in list(word_freq.items()):
            if freq <= degree:
                del word_freq[word]

    def _reconstruct_vocab(self):

        # 予約語数
        N = 6

        # buffer
        buffer_index2word = self.index2word
        buffer_word2index = self.word2index

        # 初期化
        self.index2word = {}
        self.word2index = {}

        for i in range(N):
            word = buffer_index2word[i]
            self.index2word[i] = word
            self.word2index[word] = i

        # 出現頻度の辞書の単語をインデックスに変換する辞書を作成
        # 頻出度が高い順に単語をソート
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # 辞書を再作成
        for i, (word, _) in enumerate(sorted_words):
            index = N + i # 予約後 + i
            self.word2index[word] = index
            self.index2word[index] = word

        self.vocab_size = len(self.word2index)

    def show_word_freq(self):
        # 頻出度の高い順に表示
        word_freq = self.word_freq
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            print(f"{word}: {freq}")


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

        padded_source_tokens = self.pad_sequence(source_tokens, self.max_sequence_length) # ['Hello', 'how', 'are', 'you?', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
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

        source_unique_tokens = set(source_tokens + ['<PAD>'])  # <PAD> を追加
        target_unique_tokens = set(target_tokens + ['<PAD>'])  # <PAD> を追加

        source_vocab = {token: idx + 2 for idx, token in enumerate(source_unique_tokens)}
        target_vocab = {token: idx + 2 for idx, token in enumerate(target_unique_tokens)}

        SOS_token = 0
        EOS_token = 1

        source_vocab['SOS'] = SOS_token
        source_vocab['EOS'] = EOS_token
        target_vocab['SOS'] = SOS_token
        target_vocab['EOS'] = EOS_token

        return source_vocab, target_vocab

    def tokenize(self, text):
        # Simple tokenization, split by spaces
        return text.split()

    def pad_sequence(self, tokens, max_length):
        if len(tokens) < max_length:
            padding = ['<PAD>'] * (max_length - len(tokens))
            tokens += padding
        else:
            tokens = tokens[:max_length]
        return tokens

    def tokens_to_indices(self, tokens, vocab):
        return [vocab[token] for token in tokens]


#@title Custom Dataset
class TextDataset(Dataset):
    def __init__(self, vocab, corpus, window_size):
        self.window_size = window_size
        self.vocab_size = vocab.vocab_size
        self.word2index = vocab.word2index
        self.index2word = vocab.index2word
        self.tokenized_corpora = self._create_tokenized_corpora(corpus)

    def tokenize(self, corpus):
        corpus = corpus.lower()
        return re.findall(r'\w+|[^\w\s]', corpus)

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
        corpus = self.tokenize(corpus)
        tokenized_corpus = []
        
        for word in corpus:

            if word in self.word2index:
                index = self.word2index[word] 
            else: 
                index = self.word2index['<UNK>'] # 未登録の単語として処理
                
            tokenized_corpus.append(index)
        # tokenized_corpus = [self.word2index[word] for word in corpus]
        
        return tokenized_corpus

    def tokenized_corpus2indices(self, tokenized_corpus):
        indices = []
        for word in tokenized_corpus:
            try:
                index = self.word2index[word] 
            except: 
                index = self.word2index['<UNK>'] # 未登録の単語として処理
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


    def sequence2indices(self, sequence):
        indices = []
        for word in sequence.split():
            try:
                index = self.word2index[word]
            except:
                index = self.word2index['<UNK>'] # 未登録の単語として処理
            indices.append(index)
                
        return indices

    def indices2sequence(self, indices):
        sequence = ''
        for index in indices:
            letter = self.index2word[index]
            sequence += letter
        return sequence


#@title LongSequenceDataset
class LongSequenceDataset(Dataset):
    def __init__(self, vocab, corpus, window_size, context_size):
        self.window_size = window_size
        self.context_size = context_size
        self.vocab_size = vocab.vocab_size
        self.word2index = vocab.word2index
        self.index2word = vocab.index2word
        self.tokenized_corpora = self._create_tokenized_corpora(corpus)

    def tokenize(self, corpus):
        corpus = corpus.lower()
        return re.findall(r'\w+|[^\w\s]', corpus)

    def _create_tokenized_corpora(self, corpus):
        tokenized_corpora = []
        tokenized_corpus = self._create_tokenized_corpus(corpus)
        tokenized_line = []
        sequence_size = self.window_size + 1

        # 文章をモデルの入力サイズで分割
        for i in range(0, len(tokenized_corpus) - sequence_size, self.context_size):
            tokenized_sequence = tokenized_corpus[i:i + sequence_size] #['は', '晴れ', 'です']
            tokenized_corpora.append(tokenized_sequence)

        return tokenized_corpora

    def _create_tokenized_corpus(self, corpus):
        corpus = self.tokenize(corpus)
        tokenized_corpus = []

        for word in corpus:

            if word in self.word2index:
                index = self.word2index[word]
            else:
                index = self.word2index['<UNK>'] # 未登録の単語として処理

            tokenized_corpus.append(index)
        # tokenized_corpus = [self.word2index[word] for word in corpus]

        return tokenized_corpus

    def tokenized_corpus2indices(self, tokenized_corpus):
        indices = []
        for word in tokenized_corpus:
            try:
                index = self.word2index[word]
            except:
                index = self.word2index['<UNK>'] # 未登録の単語として処理
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


    def sequence2indices(self, sequence):
        indices = []
        for word in sequence.split():
            try:
                index = self.word2index[word]
            except:
                index = self.word2index['<UNK>'] # 未登録の単語として処理
            indices.append(index)

        return indices

    def indices2sequence(self, indices):
        sequence = ''
        for index in indices:
            letter = self.index2word[index]
            sequence += letter
        return sequence

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
        self.max_sequence_length = max_sequence_length
        self.vocab = self.get_vocab(corpus)
        self.word2index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index2word = {idx:token for idx, token in enumerate(self.vocab)}
        self._add_padding_word()
        self.tokenized_corpus = self._tokenize_corpus(corpus)

    def _add_padding_word(self):
        self.vocab.append('<PAD>')
        index_pad = len(self.vocab) - 1
        self.word2index['<PAD>'] = index_pad
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
        """
        コーパスからsetコマンドで、語彙を一意に格納します。
        語彙順にソートして返却します。 
        """
        vocab = set(token for token in corpus.split())
        return sorted(vocab)

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


class AkutagawaSampleDataset(JpTextDataset):
    """
    青空文庫から芥川龍之介の作品をコーパスとしたデータセットに対応させるクラスです
    """
    def __init__(self):
        self.tagger = tagger
        self.max_sequence_length = 10
        self.base_path = '/content/whiteGPT/model/akutagawa/'
        self.word2index = self._set_obejct('word2index')
        self.index2word = self._set_obejct('index2word') 

    def _set_obejct(self, object_path):
        import pickle
        import os
        
        with open(os.path.join(self.base_path, object_path),'rb') as f:
            object = pickle.load(f)
        return object

    def _create_attention_mask(self):
        s = self.max_sequence_length
        return (torch.triu(torch.ones((s, s)),1) == 0) * 1

    def sample(self, model, corpus, n=500, t=None, use_topk=True):
        import copy
        import time
        from IPython.display import clear_output
        from IPython.core.display import display, HTML

        print(corpus, end="")

        corpus = self.tagger.parse(corpus)
        source = self.sequence2indices(corpus)
        source = source[:self.max_sequence_length]
        indices = copy.copy(source)
        # pad = self.word2index['<PAD>']
        mask = self._create_attention_mask()

        model.eval()
        
        #html = "<style>div.output_scroll { width: 100%; }</style>"
        #display(HTML(html))

        for _ in range(n):
            inputs = torch.LongTensor([source])
            outputs, _ = model(inputs, mask) 

            if use_topk:
                k = 3
                opk_values, topk_indices = torch.topk(outputs, k)
                # k個の最大値からランダムに1つをサンプリング
                topk_index = torch.randint(0, topk_indices.size(1), (1,))
                index = topk_indices[0, topk_index.item()].tolist()
            else:
                index = torch.argmax(outputs).item()

            indices.append(index)
            source.append(index)
            source = source[1:]
            text = self.indices2sequence(indices)
            #display(HTML(text))
            #clear_output(wait=True)
            print(self.index2word[index] ,end="")
            
            if t != None:
                time.sleep(t) # 視覚効果
                
    def _get_html(self):
        html = "<style>div.output_scroll { width: 100%; }</style>"
        html += "<div id=output></div>"
        html += "<script>"
        html += "function appendText(text) {"
        html += "    var outputDiv = document.getElementById('output');"
        html += "    outputDiv.innerHTML += '<p>' + text + '</p>';"
        html += "}"
        html += "</script>"
        
        return html


#@title TranslationPreTrainDataset
class TranslationPreTrainDataset(Dataset):
    def __init__(self, corpus_path, max_sequence_length):
        self.tagger = tagger
        corpus = self._read_jp_corpus(corpus_path)
        self.max_sequence_length = max_sequence_length
        self.vocab = self.get_vocab(corpus)
        self.word2index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index2word = {idx:token for idx, token in enumerate(self.vocab)}
        self._add_padding_word()
        self.tokenized_corpus = self.tokenize_corpus(corpus)

    def _add_padding_word(self):
        self.vocab.append('<PAD>')
        index_pad = len(self.vocab) - 1
        self.word2index['<PAD>'] = index_pad
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


#@title ClassifierDataset  (textual entailment)
class ClassifierDataset(Dataset):

    def __init__(self, vocab, sentences, context_size=15, tagger=None):
        self.vocab_size = vocab.vocab_size
        self.word2index = vocab.word2index
        self.index2word = vocab.index2word
        self.max_sequence_length = context_size
        self.tagger = tagger
        self.pairs = self._make_pairs(sentences)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        source, target = self.pairs[idx]

        try:
            source = torch.LongTensor(source)
            target = torch.LongTensor([target])
        except:
            print('source', source)
            print('target', target)

        return {
            'source': source,
            'target': target
        }

    def _make_pairs(self, sentences):

        delimiter = ':' # または '\t' など
        pairs = []

        for sentence in sentences:
            QA = sentence.split(delimiter)
            Q = self._Q( QA[0] )
            A = self._A( QA[1] )
            pairs.append([Q, A])

        return pairs

    def _Q(self, Q):
        indices = self.sentence2indices(Q)
        indices = self._pad_sequence(indices)
        return indices

    def _A(self, A):
        # 含意
        if A == '否定':
            A = 0
        elif A == '肯定':
            A = 1
        else: # '中立'
            A = 2

        return A

    def load_classifier_data(self, data_path):
        #lines = open(data_path, encoding='utf-8').read().strip().split('\n')
        pass


    def _pad_sequence(self, indices):
        max_length = self.max_sequence_length

        # 文字数がwindowサイズよりも少ない場合の処理
        if len(indices) < max_length:
            index_pad = self.word2index['<PAD>']
            padding = [index_pad] * max_length
            indices += padding

        return indices[:max_length]

    def _tokenize(self, sentence):
        if self.tagger:
            sentence = self.tagger.parse(sentence)

        tokens = sentence.split()

        return tokens

    def sentence2indices(self, sentence):

        tokens = self._tokenize(sentence)

        # return [self.word2index[token] for token in tokens]

        # 未知の単語にUnknownを割り当てる処理
        """
        indices = []
        for token in tokens:
            if token in vocab.word2index:
                index = self.word2index[token]
            else:
                index =self.word2index['<UNK>']
            indices.append(index)
        return indices
        """
        index_unk = self.word2index['<UNK>']

        return   [self.word2index.get(token, index_unk) for token in tokens]

    def indices2sentence(self, indices):
        tokens = [self.index2word[index] for index in indices]
        sentence = ''.join(tokens)
        return sentence

    """
    From JpTextDataset class
    """
    def sequence2indices(self, sequence):
        return self.sentence2indices(sequence)

    def indices2sequence(self, indices):
        return self.indices2sentence(indices)
