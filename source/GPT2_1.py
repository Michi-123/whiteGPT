# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


""" GPTモデル """

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

    def forward(self, q, k, v, casual_mask=None, padding_mask=None):
        score = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_d_k
        if casual_mask is not None:
            # infは数値ではないのでプログラムによってはエラーを起こします。
            score = score.masked_fill(casual_mask == 1, -1e9)

        if padding_mask is not None:
            score = score.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == 1, -1e9)


        attn_weights = F.softmax(score, dim=-1)
        # 特定の単語に注意を払いすぎないようにdropoutを適用します
        attn_weights = self.dropout(attn_weights)
        attention_output = torch.matmul(attn_weights, v)

        return attention_output, attn_weights


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

    def forward(self, q, k, v, past=None, casual_mask=None, padding_mask=None):
        N = q.size(0) # バッチサイズ
        qS = q.size(1) # ウィンドウサイズ
        kS = k.size(1)
        vS = v.size(1)
        H = self.n_head # マルチヘッドの数
        D = self.d_model // self.n_head # 潜在空間の次元。（Cross-Attentionの場合、個別に定義します）

        # 線形変換
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        # 展開
        q = q.view(N, qS, H, D)
        k = k.view(N, kS, H, D)
        v = v.view(N, vS, H, D)

        # 転置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # past処理
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        # Scaled dot-product attention
        x, w = self.attention(q, k, v, casual_mask, padding_mask)

        # 転置
        x = x.transpose(1, 2)

        # 結合
        x = x.contiguous()

        # 展開
        x = x.view(N, qS, -1)

        # 線形変換
        x = self.fc(x)

        # 正則化
        x = self.dropout(x)

        present = torch.stack([k, v])

        return x, present, w


#@title TransformerBlock
class TransformerBlock(nn.Module):
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
    def forward(self, x, past=None, casual_mask=None, padding_mask=None):
        _x = x
        x = self.norm_1(x)
        x, present, w = self.attn(x, x, x, past=past, casual_mask=casual_mask, padding_mask=padding_mask)
        x = x + _x

        # Residual x
        _x = x

        # Feed Forward
        x = self.norm_2(x)
        x = self.ff(x)
        x = x + _x

        return x, present, w


#@title GPT2
class GPT2(nn.Module):
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


    def forward(self, x, past=None, casual_mask=casual_mask, padding_mask=padding_mask):
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
            x, present, w = block(x, past=past_block, casual_mask=casual_mask, padding_mask=padding_mask)
            presents.append(present)

        # 正規化(GPT-2仕様)
        x = self.norm(x)

        x = x.view(-1, self.context_size * self.d_model)

        # 線形変換
        logits = self.fc(x)

        return logits, torch.stack(presents, dim=1), w