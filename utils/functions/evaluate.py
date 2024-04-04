# @title Evaluate

import copy
import torch

from IPython.display import HTML, display

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
        pad = self.dataset.word2index['<PAD>']

        source = self.dataset.sequence2indices(corpus)
        source = source[:self.context_size]
        indices = copy.copy(source)

        for i in range(max_token_size):
            inputs = [source + [pad] * (self.context_size - len(source))]
            inputs = torch.LongTensor(inputs).to(device)

            outputs ,_ = model(inputs, mask)
            if 1:
                index = torch.argmax(outputs).item()
            else:
                k = 3
                opk_values, topk_indices = torch.topk(outputs, k)
                # k個の最大値からランダムに1つをサンプリング
                topk_index = torch.randint(0, topk_indices.size(1), (1,))
                index = topk_indices[0, topk_index.item()].tolist()

            indices.append(index)
            # source = torch.cat([source, torch.LongTensor([index]).unsqueeze(0)], dim=1)
            source.append(index)
            # source.pop(0)
            source = source[1:]

            print(self.dataset.index2word[index] ,end="")
            
            
        # self.dataset.indices2sequence(indices)


    def input_tokens(self, corpus):
        pad = self.dataset.word2index["<PAD>"]
        source = self.dataset.sequence2indices(corpus)
        source = source[:self.context_size]
        inputs = [source + [pad] * (self.context_size - len(source))]
        inputs = torch.LongTensor(inputs).to(device)
        return inputs
        

if __name__ == "__main__": 
    pass
