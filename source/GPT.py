# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR

""" GPTモデル """

#@title create_attention_mask
def create_attention_mask(seq_length):
    # 全ての要素が1の行列を作成
    attention_mask = torch.ones((seq_length, seq_length))

    # 対角線より上を0にする
    attention_mask = torch.triu(attention_mask, diagonal=1)

    attention_mask = attention_mask == 0
    return attention_mask * 1


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
        self.sqrt_d_k = d_model ** 0.5 # sqrt(d_k)　と同じ
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) /  self.sqrt_d_k

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn

#@title  Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.fc_q = nn.Linear(d_model, d_model * n_head)
        self.fc_k = nn.Linear(d_model, d_model * n_head)
        self.fc_v = nn.Linear(d_model, d_model * n_head)
        self.attn = ScaledDotProductAttention(d_model)
        self.fc = nn.Linear(n_head * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        N = q.size(0) # バッチサイズ（Transformerの場合、QKVは同じサイズ）
        S = q.size(1) # ウィンドウサイズ（Transformerの場合、QKVは同じサイズ）
        H = self.n_head # マルチヘッドの数
        D = self.d_model # 潜在区間の次元（Cross Attentonの場合、個別に定期）

        # 線形変換
        q = self.fc_q(q).view(N, S, H, D)
        k = self.fc_k(k).view(N, S, H, D)
        v = self.fc_v(v).view(N, S, H, D)

        # Scaled dot-product attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attn(q, k, v, mask=mask)

        # 結合
        q = q.transpose(1, 2).contiguous().view(N, S, -1)

        # 線形変換
        q = self.fc(q)
        
        q = self.dropout(q)

        return q, attn


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
        self.attn = MultiHeadAttention(n_head, d_model)
        self.ff = FeedForward(d_model)

        # Initialize gamma (weight) with N(0, 0.02)
        nn.init.normal_(self.norm_1.weight, mean=0, std=0.02)
        nn.init.normal_(self.norm_2.weight, mean=0, std=0.02)

    # GPT-1
    def _forward(self, x, mask=None):
        residual_x = x
        x, w = self.attn(x, x, x, mask)
        x = self.norm_1(x + residual_x)

        residual_x = x
        x = self.ff(x)
        x = self.norm_2(x + residual_x)

        return x, w

    # GPT-2
    def forward(self, x, mask=None):
        _x = x
        x = self.norm_1(x)
        x, w = self.attn(x, x, x, mask)
        
        _x = x + _x

        x = self.norm_2(_x)
        x = self.ff(x) + _x

        return x, w

#@title GPT
class GPT(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, n_head, n_block):
        super(GPT, self).__init__()
        
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.n_block = n_head
        self.n_block = n_block
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # self.position_embedding = PositionEmbedding(context_size, d_model)
        self.positional_encoding = PositionalEncoding(context_size, d_model)
        self.dropout = nn.Dropout(0.1)
        self.transformer_block = nn.ModuleList([TransformerBlock(d_model, n_head) for _ in range(self.n_block)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model * context_size, vocab_size)
        
        init.xavier_uniform_(self.fc.weight)        
        init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        init.normal_(self.position_embedding.embedding.weight, mean=0.0, std=0.02)


    def forward(self, x, mask=None):
        # 埋め込み
        x = self.token_embedding(x) + self.positional_encoding(x)
        x = self.dropout(x)

        # Transformer ブロック
        for block in self.transformer_block:
            x, w = block(x, mask)
        
        # 正規化(GPT-2仕様)
        x = self.norm(x)
        
        x = x.view(-1, self.context_size * self.d_model)
        
        # 線形変換
        x = self.fc(x)

        return x, w
