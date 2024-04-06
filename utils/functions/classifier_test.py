# @title Test
import torch
def test(model, mask, classifier, vocab, dataloader,):
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