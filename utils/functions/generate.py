# @title Generate
def genarate(source_indices, target_indices, outputs):
    # INPUT
    # 辞書の取得
    my_dict = dataset.word2index
    # dict_keysオブジェクトをリストに変換
    keys_list = list(my_dict.keys())
    # テキストデータの検証
    text = ""
    for index in source_indices[0]:
        # リストからキーを取得
        key_at_index = keys_list[index]
        text += key_at_index + " "
    print('source', text)

    # OUTPUT
    output_index = outputs[0].argmax().item()

    # dict_keysオブジェクトをリストに変換
    keys_list = list(my_dict.keys())
    # テキストデータの検証
    # リストからキーを取得
    print('output',keys_list[output_index])
    print('target', keys_list[target_indices[0]])
    print()

if __name__ == "__main__": 
    pass
