# @title Evaluate

import copy
import torch

from IPython.display import HTML, display

from whiteGPT.source.GPT import create_pad_mask

""" 改行処理 """
def set_css(*args, **kwargs):
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
        self.tagger = None

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


    def generate(self, corpus, model, mask=None, max_token_size=500, eos='<EOS>'):
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
            
            next_word = self.dataset.index2word[index]

            print(next_word ,end="")
            
            if next_word == eos:
                break
            
  
    def generate2(self, corpus, model, mask=None, max_token_size=500, topk=0, eos='<EOS>'):
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

            outputs ,_, _ = model(inputs, None, mask)
            if topk == 0:
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
            
            next_word = self.dataset.index2word[index]

            print(next_word, end="")
            
            if next_word == eos:
                break


    def generate_fine_tuned(self, sentence, model, mask, max_token_size=500, top_k=0, top_p=0, eos='<EOS>'):
        model.eval()
        model.cpu()
        pad = self.dataset.word2index['<PAD>']

        parsed_sentence = self.tagger.parse(sentence)
        source = self.dataset.sequence2indices(parsed_sentence)
        source = source[:self.context_size]

        for i in range(max_token_size):
            if len(source) <= self.context_size:
                inputs = source + [pad] * self.context_size
                inputs = inputs[:self.context_size]
            else:
                source = source[1:]
                inputs = source
            # print(self.dataset.indices2sequence(inputs))
            inputs = torch.LongTensor([inputs]).cpu()

            outputs ,_, _ = model(inputs, None, mask)
            if top_k == 0:
                index = torch.argmax(outputs).item()
            elif top_k > 0:

                topk_values, topk_indices = torch.topk(outputs, top_k)

                # topk_values をSoftmax で確率分布に変換
                topk_values = torch.softmax(topk_values, dim=-1)
                
                # top-p で切り捨て:Handle the case where no values are greater than top_p
                valid_indices = topk_values > top_p
                if valid_indices.any():
                    topk_indices = topk_indices[valid_indices]

                    # k個の最大値からランダムに1つをサンプリング
                    topk_index = torch.randint(0, topk_indices.size(0), (1,)) # Change size(1) to size(0) to get the first dimension
                    index = topk_indices[topk_index.item()].tolist()
                else:
                    # If no values are greater than top_p, take the most probable index (index 0)
                    index = topk_indices[0, 0].item()  # Access the first element of the first dimension
                
            source.append(index)
            
            next_word = self.dataset.index2word[index]

            print(next_word, end="")

            if next_word == eos:
                break

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


def classifier_test2(model, mask, classifier, vocab, dataloader,):
    index2word = vocab.index2word
    model.eval()
    answers = {0:'否定', 1:'肯定', 2:'中立' }

    count_ok = 0
    past = None
    
    for i, batch in enumerate(dataloader):
        # print(i)
        source = batch['source']
        target = batch['target']

        # 事前学習モデルからlogitsを取得
        x, past, _ = model(source, past, mask)

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
