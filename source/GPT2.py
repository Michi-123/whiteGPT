# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .GPT import TransformerBlock, PositionalEncoding, PositionEmbedding, GPT


""" GPTモデル """

#@title TransformerBlock
class TransformerBlock(TransformerBlock):

    # GPT-2
    def forward(self, x, past=None, mask=None):

        _x = x
        x = self.norm_1(x)

        q, k, v = x, x, x
        present = q

        if past is not None:
            k = k + past
            v = v + past

        x, w = self.attn(q, k, v, mask)
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
        self.n_block = n_head
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
        for block in self.transformer_block:
            x, past, w = block(x, past, mask)

        # 正規化(GPT-2仕様)
        x = self.norm(x)

        x = x.view(-1, self.context_size * self.d_model)

        # 線形変換
        x = self.fc(x)

        return x, past, w