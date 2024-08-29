import re
import torch
from torch.utils.data import DataLoader

   
#@title create_casual_mask
def create_casual_mask(seq_length):
    # 全ての要素が1の行列を作成
    attention_mask = torch.ones((seq_length, seq_length))
    # 対角線より上を0に変換
    attention_mask = torch.triu(attention_mask, diagonal=1)
    # True/Falseに変換
    attention_mask = attention_mask == 1

    return attention_mask


#@title create_padding_mask
def create_padding_mask(sequence):
    mask = (sequence == 0)
    return mask
    
    
def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

#@title create_padding_mask
def create_padding_mask(sequence):
    mask = (sequence == 0)
    return mask

def _pad_collate(batch):
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        src_list.append(_src)
        tgt_list.append(_tgt)

    src_pad = nn.utils.rnn.pad_sequence(src_list, padding_value=0, batch_first=True)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_list, padding_value=0, batch_first=True)

    return src_pad, tgt_pad



#@title make_input_tensor
def make_input_tensor(batch_size, max_seq_length):
    sos = 1
    sequence_size = max_seq_length

    # <sos>トークンを最初の要素とする1次元テンソルを生成
    sos_tensor = torch.ones(batch_size, dtype=torch.long) * sos
    # <pad>トークンを(batch_size, sequence_size-1)のテンソルとして生成
    pad_tensor = torch.zeros(batch_size, sequence_size-1, dtype=torch.long)
    # <sos>トークンと<pad>トークンを結合
    input_tensor = torch.cat([sos_tensor.unsqueeze(1), pad_tensor], dim=1)

    return input_tensor
    
    
    
    
    
import re
import torch
from torch.utils.data import DataLoader

# @title test_from_dataset
def test_from_dataset(model, dataset):
    # デバイスの設定
    device = ('cpu')

    vocab_en = dataset.vocab_en
    vocab_ja = dataset.vocab_ja

    PAD = 0
    EOS = 2
    model.eval()
    model.to(device)

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        try:
            for i, (src, tgt) in enumerate(test_dataloader):
                src = src.to(device)
                tgt = tgt.to(device)

                context_size = src.size(1)

                casual_tgt_mask = create_casual_mask(context_size).to(device)
                tgt_padding_mask = create_padding_mask(tgt).to(device)
                src_padding_mask = create_padding_mask(src).to(device)

                tgt_input = None
                output = model(src, tgt_input, casual_tgt_mask, tgt_padding_mask, src_padding_mask)

                # 英語
                print('原文:', end='')
                src_indicies = src.squeeze(1).squeeze(0)
                src_indicies = src_indicies.tolist()
                for index in src_indicies:
                    if index == PAD:
                        break
                    token = vocab_en.index2word[index]
                    print(token, end=" ")
                print()

                # 正解データ
                print('正解:', end='')
                tgt_indices = tgt.squeeze(1).squeeze(0)
                tgt_indices = tgt_indices.tolist()
                for index in tgt_indices:
                    if index == EOS:
                        break
                    token = vocab_ja.index2word[index]
                    print(token, end="")
                print()

                # 生成された日本語
                print('推論:', end='')
                output = output.squeeze(1)[0]
                output_indices = output.argmax(dim=1)
                output_indices = output_indices.tolist()
                for index in output_indices:
                    if index == EOS:
                        break
                    token = vocab_ja.index2word[index]
                    print(token, end="")

                print('\n')

                if (i+1) % 3 == 0:
                    input('next')
        except:
            print('Error.')


#@title test_from_human_input
def test_from_human_input(model, context_size):
    PAD = 0
    SOS = 1
    EOS = 2
    
    device = 'cpu'
    model.eval()
    model.to(device)

    sentence = input('英文')
    sequence = normalizeString(sentence)
    indices = []
    for word in sequence.split():
        try:
            index = vocab_en.word2index[word]
            indices.append(index)
        except:
            print('単語がありません')
            return

    indices = indices + [0] * context_size
    indices = indices[:context_size]

    input_tensor = torch.LongTensor(indices).unsqueeze(0)

    tgt_casual_mask = create_casual_mask(context_size)

    with torch.no_grad():
        src = input_tensor.to(device)

        tgt_casual_mask = create_casual_mask(context_size).to(device)
        tgt_padding_mask = None
        src_padding_mask = create_padding_mask(src).to(device)

        tgt_input = None
        output = model(src, tgt_input, tgt_casual_mask, tgt_padding_mask, src_padding_mask)

        # 生成された日本語
        print('推論:', end='')
        output = output.squeeze(1)[0]
        output_indices = output.argmax(dim=1)
        output_indices = output_indices.tolist()
        for index in output_indices:
            if index == EOS:
                break
            token = vocab_ja.index2word[index]
            print(token, end="")

        print('\n')

def normalizeString(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([.!?])", r" \1", sentence)
    sentence = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", sentence)
    return sentence
    
