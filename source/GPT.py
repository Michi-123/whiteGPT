# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

""" GPTモデル """

#@title create_attention_mask
def create_attention_mask(seq_length):
    # 全ての要素が1の行列を作成
    attention_mask = torch.ones((seq_length, seq_length))

    # 対角線より上を0にする
    attention_mask = torch.triu(attention_mask, diagonal=1)

    attention_mask = attention_mask == 0
    return attention_mask * 1

#@title create_pad_mask
def create_pad_mask(source):
    # True -> 1; False -> 0
    source = source.ne(0) * 1
    sq_masks = []
    for tokens in source:
        rows = []
        for n in range(1, len(tokens) + 1):
            mask = torch.cat((tokens[:n], tokens[n:] * 0), dim=0)
            rows.append(mask)
        sq_mask = torch.stack(rows)
        sq_masks.append(sq_mask)

    pad_mask = torch.stack(sq_masks).unsqueeze(1)

    return pad_mask

#@title add_random_pad
import random
pad_index = 0 # dataset.word2index['<PAD>']
def add_random_pad(data, epoch):
    # パディングする位置（教師データを除く位置まで）
    source_size = 1
    n = random.randint(1, context_size - 1)
    # 次のトークンを教師データに設定
    source = data[:, :-1].clone()
    target = data[:,n + 1].clone()
    # 入力元をパディング
    source[:, n+1:] = pad_index
    return source, target


#@title PositionEmbedding
class PositionEmbedding(nn.Module):
    def __init__(self, context_size , d_model):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(context_size, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device)
        return self.embedding(positions)


#@title PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, context_size, d_model):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (context_size, d_model) with positional encodings
        pe = torch.zeros(context_size, d_model)

        # for pos in range(context_size):
        #     for i in range(d_model):
        #         if  i % 2 == 0:
        #             pe[pos,i] = math.sin(pos/(10000**((2*i)/d_model)))
        #         else:
        #             pe[pos,i] = math.cos(pos/(10000**((2*(i-1))/d_model)))

        for pos in range(context_size):
            for i in range(0, d_model, 2):
                pe[pos,i]   = math.sin(pos/(10000**((2*i)/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**((2*i)/d_model)))

        # 学習パラメーターの更新対象から外してクラス変数に確保(重要)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # positional encodingを埋め込みベクトルへ追加します
        return self.pe[:, :x.size(1)].detach()


# @title ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1):
        super().__init__()
        self.sqrt_d_k = d_model ** 0.5 # sqrt(d_k)と同じ
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        score = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_d_k

        if mask is not None:
            # infは数値ではないのでプログラムによってはエラーを起こします。
            #score = score.masked_fill(mask == 0, float("-inf"))
            score = score.masked_fill(mask == 0, -1e9) 

        attn_weights = F.softmax(score, dim=-1)
        # 特定の単語に注意を払いすぎないようにAttention scoreにもdropoutを適用します
        attn_weights = self.dropout(attn_weights)
        attention_output = torch.matmul(attn_weights, v)

        return attention_output, attn_weights

#@title  Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.fc_q = nn.Linear(d_model, d_model) # d_model * n_head // n_head
        self.fc_k = nn.Linear(d_model, d_model) # d_model * n_head // n_head
        self.fc_v = nn.Linear(d_model, d_model) # d_model * n_head // n_head
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.fc = nn.Linear(d_model, d_model) # d_model * n_head // n_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        N = q.size(0) # バッチサイズ
        S = q.size(1) # ウィンドウサイズ
        H = self.n_head # マルチヘッドの数
        D = self.d_model // self.n_head # 潜在空間の次元。（Cross-Attentionの場合、個別に定義します）

        # 線形変換
        q = self.fc_q(q).view(N, S, H, D)
        k = self.fc_k(k).view(N, S, H, D)
        v = self.fc_v(v).view(N, S, H, D)

        # 展開
        q = q.view(N, S, H, D)
        k = k.view(N, S, H, D)
        v = v.view(N, S, H, D)

        # 転置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        x, attn_weights = self.attention(q, k, v, mask=mask)

        # 転置
        x = x.transpose(1, 2)
        
        # 結合
        x = x.contiguous()
        
        # 展開
        x = x.view(N, S, -1)

        # 線形変換
        x = self.fc(x)

        # 正則化
        x = self.dropout(x)

        return x, attn_weights


#@title FeedForward
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model * 4 )
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


#@title TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_head, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)

        # Initialize gamma (weight) with N(0, 0.02)
        nn.init.normal_(self.norm_1.weight, mean=0, std=0.02)
        nn.init.normal_(self.norm_2.weight, mean=0, std=0.02)

    # GPT-1
    def forward(self, x, mask=None):
        residual_x = x
        x, w = self.attn(x, x, x, mask)
        x = self.norm_1(x + residual_x)

        residual_x = x
        x = self.ff(x)
        x = self.norm_2(x + residual_x)

        return x, w


#@title GPT-Original
class GPT(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, n_head, n_block, dropout=0.1):
        super(GPT, self).__init__()
        
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_block = n_block
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # self.position_embedding = PositionEmbedding(context_size, d_model)
        self.positional_encoding = PositionalEncoding(context_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_block = nn.ModuleList([TransformerBlock(d_model, n_head, dropout) for _ in range(self.n_block)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model * context_size, vocab_size)
        
        init.xavier_uniform_(self.fc.weight)        
        init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # init.normal_(self.position_embedding.embedding.weight, mean=0.0, std=0.02)


    def forward(self, x, mask=None):
        # 埋め込み
        x = self.token_embedding(x) + self.positional_encoding(x)
        x = self.dropout(x)

        # Transformer ブロック
        for block in self.transformer_block:
            x, w = block(x, mask)
      
        x = x.view(-1, self.context_size * self.d_model)
        
        # 線形変換
        x = self.fc(x)

        return x, w