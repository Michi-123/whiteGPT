# @title Evaluate

import copy
import torch

from IPython.display import HTML, display

from whiteGPT.source.GPT import create_pad_mask

""" 改行処理 """
def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluate:

    def __init__(self, dataset, context_size):
        self.dataset = dataset
        self.context_size = context_size

    def predict(self, source_indices, target_indices, outputs):
        # INPUT
        # 辞書の取得
        my_dict = self.dataset.word2index
        # dict_keysオブジェクトをリストに変換
        keys_list = list(my_dict.keys())
        # テキストデータの検証
        text = ""
        for index in source_indices[0]:
            # リストからキーを取得
            key_at_index = keys_list[index]
            text += key_at_index + " "
        print('source', text)

        output_index = outputs[0].argmax().item()
        target_index = target_indices[0]

        # dict_keysオブジェクトをリストに変換
        keys_list = list(my_dict.keys())
        # テキストデータの検証
        # リストからキーを取得
        print('target', keys_list[target_index])
        print('output', keys_list[output_index])
        print()


    def generate(self, corpus, model, mask=None, max_token_size=500):
        model.eval()
        model.cpu()
        pad = self.dataset.word2index['<PAD>']

        source = self.dataset.sequence2indices(corpus)
        source = source[:self.context_size]
        #indices = copy.copy(source)

        for i in range(max_token_size):
            inputs = [source + [pad] * (self.context_size - len(source))]
            inputs = torch.LongTensor(inputs).cpu()
            
            if mask is not None:
                mask = create_pad_mask(inputs)

            outputs ,_ = model(inputs, mask)
            if 1:
                index = torch.argmax(outputs).item()
            else:
                k = 3
                opk_values, topk_indices = torch.topk(outputs, k)
                # k個の最大値からランダムに1つをサンプリング
                topk_index = torch.randint(0, topk_indices.size(1), (1,))
                index = topk_indices[0, topk_index.item()].tolist()

            #indices.append(index)
            source.append(index)
            source = source[1:]

            print(self.dataset.index2word[index] ,end="")
  

    def input_tokens(self, corpus):
        pad = self.dataset.word2index["<PAD>"]
        source = self.dataset.sequence2indices(corpus)
        source = source[:self.context_size]
        inputs = [source + [pad] * (self.context_size - len(source))]
        inputs = torch.LongTensor(inputs).to(device)
        return inputs
        
    def generate_long(self, corpus, model, d_model, mask=None, max_token_size=500, deterministic_select=True):
        model.eval()
        model.cpu()
        pad = self.dataset.word2index['<PAD>']

        source = self.dataset.sequence2indices(corpus)

        latent = torch.zeros(1, self.context_size, d_model, dtype=torch.float).cpu()

        for i in range(len(source) - self.context_size - 1):
            inputs = source[i:i + self.context_size]
            inputs = torch.LongTensor(inputs).cpu()
            _outputs, latent, _w = model(inputs, latent, mask)


        inputs = source[-self.context_size:]
        for i in range(max_token_size):
            inputs = torch.LongTensor(inputs).cpu()

            # 推論
            outputs, latent, w = model(inputs, latent, mask)

            if deterministic_select:
                index = torch.argmax(outputs).item()
            else:
                k = 3
                opk_values, topk_indices = torch.topk(outputs, k)
                # k個の最大値からランダムに1つをサンプリング
                topk_index = torch.randint(0, topk_indices.size(1), (1,))
                index = topk_indices[0, topk_index.item()].tolist()

            #indices.append(index)
            source.append(index)
            source = source[1:]

            print(self.dataset.index2word[index] ,end="")


# @title Test
import torch
def classifier_test(model, mask, classifier, vocab, dataloader,):
    index2word = vocab.index2word
    model.eval()
    answers = {0:'否定', 1:'肯定', 2:'中立' }

    count_ok = 0
    
    for i, batch in enumerate(dataloader):
        # print(i)
        source = batch['source']
        target = batch['target']

        # 事前学習モデルからlogitsを取得
        x, _ = model(source, mask)

        # 分類ヘッドの追加
        logits = classifier(x)
        
        pred_index = torch.argmax(logits[0]).squeeze().item()

        for idx in source[0]:
            word = index2word[idx.item()]
            if word == '<PAD>': break
            print(word ,end="")

        target_index = target[0].item()

        if pred_index == target_index:
            result = 'OK'
            count_ok += 1
        else:
            result = 'NG'

        print(' -->', answers[pred_index], answers[target_index], result)
    
    print('OK: {}/{}'.format(count_ok, i+1))




if __name__ == "__main__": 
    pass
