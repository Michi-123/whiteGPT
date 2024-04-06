# @title Test
def test(model, mask, vocab, dataloader,):
    index2word = vocab.index2word
    model.eval()
    answers = ['肯定','否定', '中立' ]

    for i, batch in enumerate(dataloader):
        # print(i)
        source = batch['source']
        target = batch['target']

        # 事前学習モデルからlogitsを取得
        x, _ = model(source, mask)

        # 分類ヘッドの追加
        logits = classifier(x)
        index = torch.argmax(logits[0]).squeeze().item()

        for id in source[0]:
            token = index2word[id.item()]
            print(token ,end="")
        
        result = 'OK' if index == target[0].item() else 'NG'
        print(' -->', answers[index], answers[target[0].item()])