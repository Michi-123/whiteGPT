# -*- coding: utf-8 -*-
"""# ライブラリーのインポート"""

#@title import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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
                pe[pos,i]   = math.sin(pos/(10000**(i/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**(i/d_model)))

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

    def forward(self, q, k, v,  casual_mask=None, padding_mask=None):
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

        return x, w


# @title Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, max_seq_length, num_layers, d_model, n_head, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model)
        self.layers = nn.ModuleList([EncoderLayer( d_model, n_head, dropout) for _ in range(num_layers)])

    def forward(self, x, casual_mask=None, padding_mask=None):
        x = self.encoder_embedding(x) + self.positional_encoding(x)
        x = x * math.sqrt(self.d_model)

        for layer in self.layers:
            x, w = layer(x, casual_mask=None, padding_mask=padding_mask)

        return x


#@title EncoderLayer
class EncoderLayer(nn.Module):
    # GPT-2
    def __init__(self, d_model, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        # Initialize gamma (weight) with N(0, 0.02)
        nn.init.normal_(self.norm_1.weight, mean=0, std=0.02)
        nn.init.normal_(self.norm_2.weight, mean=0, std=0.02)

    # GPT-2
    def forward(self, x, casual_mask=None, padding_mask=None):
        _x = x
        x = self.norm_1(x)
        x, w = self.self_attn(x, x, x, casual_mask=None, padding_mask=padding_mask)
        x = x + _x

        # Residual x
        _x = x

        # Feed Forward
        x = self.norm_2(x)
        x = self.ff(x)
        x = x + _x

        return x, w


# @title Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, context_size, n_block, d_model, n_head, dropout=0.1):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_block = n_block

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(context_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, dropout) for _ in range(self.n_block)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, casual_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        x = self.token_embedding(tgt) + self.positional_encoding(tgt)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)

        for layer in self.layers:
            x, _ = layer(x, memory, casual_mask=casual_mask, tgt_padding_mask=tgt_padding_mask, memory_padding_mask=memory_padding_mask)

        x = self.norm(x)
        logits = self.fc(x)
        return logits


#@title DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        # Initialize gamma (weight) with N(0, 0.02)
        nn.init.normal_(self.norm_1.weight, mean=0, std=0.02)
        nn.init.normal_(self.norm_2.weight, mean=0, std=0.02)
        nn.init.normal_(self.norm_3.weight, mean=0, std=0.02)

    # GPT-2
    def forward(self, x, memory, casual_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        # Residual x
        _x = x
        x = self.norm_1(x)
        x, _ = self.self_attn(x, x, x, casual_mask=casual_mask, padding_mask=tgt_padding_mask)
        x = x + _x

        # Residual x
        _x = x

        x = self.norm_2(x)
        x, _ = self.cross_attn(x, memory, memory, casual_mask=None, padding_mask=memory_padding_mask)
        x = x + _x

        # Residual x
        _x = x

        # Feed Forward
        x = self.norm_3(x)
        x = self.ff(x)
        x = x + _x

        return x, None


# @title Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, num_encoder_layers, num_decoder_layers, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, max_seq_length, num_encoder_layers, d_model, n_head, dropout=dropout)
        self.decoder = Decoder(tgt_vocab_size, max_seq_length, num_decoder_layers, d_model, n_head, dropout=dropout)
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_length = max_seq_length

    def forward(self, src, tgt, tgt_casual_mask=None, tgt_padding_mask=None, src_padding_mask=None):
        memory = self.encoder(src, casual_mask=None, padding_mask=src_padding_mask)

        input_tensor = make_input_tensor(src.size(0), src.size(1)) # [2,15]
        input_tensor = input_tensor.to(src.device)
        # 一度に一文を推論
        output = self.decoder(input_tensor, memory, casual_mask=tgt_casual_mask, tgt_padding_mask=tgt_padding_mask, memory_padding_mask=src_padding_mask)

        return output
        