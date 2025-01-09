# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .GPT import FeedForward, TransformerBlock, PositionalEncoding, PositionEmbedding, ScaledDotProductAttention, GPT


""" GPTモデル """

#@title  Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 全結合層をザビエル方式で初期化
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, q, k, v, past=None, mask=None):
        N = q.size(0) # バッチサイズ
        S = q.size(1) # ウィンドウサイズ
        H = self.n_head # マルチヘッドの数
        D = self.d_model // self.n_head # 潜在空間の次元。（Cross-Attentionの場合、個別に定義します）

        # 線形変換
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        # 展開
        q = q.view(N, S, H, D)
        k = k.view(N, S, H, D)
        v = v.view(N, S, H, D)

        # 転置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # past処理
        if past is not None:
            pk, pv = past[0], past[1]  # 過去のキーと値
            
            # 過去と現在を平均化
            k = (pk + k) / 2  # [N, H, S, D]
            v = (pv + v) / 2  # [N, H, S, D]

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

        present = torch.stack([k, v])

        return x, present, attn_weights

#@title TransformerBlock
class TransformerBlock(nn.Module):
    # GPT-2
    def __init__(self, d_model, n_head, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_head, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        # Initialize gamma (weight) with N(0, 0.02)
        nn.init.normal_(self.norm_1.weight, mean=0, std=0.02)
        nn.init.normal_(self.norm_2.weight, mean=0, std=0.02)

    # GPT-2
    def forward(self, x, past=None, mask=None):

        _x = x
        x = self.norm_1(x)

        x, present, w = self.attn(x, x, x, past=past, mask=mask)
        x = x + _x

        # Residual x
        _x = x

        # Feed Forward
        x = self.norm_2(x)
        x = self.ff(x)
        x = x + _x

        return x, present, w


#@title GPT2
class GPT2(GPT):
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


    def forward(self, x, past=None, mask=None):
        # 埋め込み
        x = self.token_embedding(x) + self.positional_encoding(x)
        x = self.dropout(x)

        # Transformer ブロック
        presents = []
        if past is not None:
            past = torch.unbind(past, dim=1)
        else:
            past = [None] * self.n_block

        for block, past_block in zip(self.transformer_block, past):
            x, present, w = block(x, past=past_block, mask=mask)
            presents.append(present)

        # 正規化(GPT-2仕様)
        x = self.norm(x)

        x = x.view(-1, self.context_size * self.d_model)

        # 線形変換
        logits = self.fc(x)

        return logits, torch.stack(presents, dim=1), w