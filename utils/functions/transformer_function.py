#@title functions
import torch

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)

    return tgt_mask, src_padding_mask, tgt_padding_mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _pad_collate(batch):
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        src_list.append(_src)
        tgt_list.append(_tgt)

    src_pad = nn.utils.rnn.pad_sequence(src_list, padding_value=0, batch_first=True)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_list, padding_value=0, batch_first=True)

    return src_pad, tgt_pad
