#@title def visualize_attention_weights(attention_weights, parsed_corpus):

# !apt-get -y install fonts-ipafont-gothic
# !pip install japanize-matplotlib

#import japanize_matplotlib
import matplotlib.pyplot as plt
import torch

def visualize_attention_weights(attention_weights, parsed_corpus):
    attention_weights = attention_weights.detach()

    # attention_weights: shape (num_heads, sequence_length, sequence_length)
    num_heads, context_size, _ = attention_weights.shape

    words = parsed_corpus.split()[:context_size]

    # グラフの作成
    if num_heads == 1:
        fig, axs = plt.subplots(figsize=(10, 6))
        axs = [axs]  # Encapsulate the single axis in a list to unify the handling below
    else:
        fig, axs = plt.subplots(num_heads, 1, figsize=(10, 6 * num_heads))


    # 各ヘッドごとに重みを可視化
    for i in range(num_heads):
        # x軸、y軸のラベルに単語を設定
        axs[i].imshow(attention_weights[i], cmap='hot', interpolation='nearest')
        axs[i].set_title(f'Head {i+1} Attention')
        axs[i].set_xticks(torch.arange(len(words)))
        axs[i].set_xticklabels(words, rotation=45, ha='right')
        axs[i].set_yticks(torch.arange(len(words)))
        axs[i].set_yticklabels(words)
        axs[i].set_xlabel('Key')
        axs[i].set_ylabel('Query')
        axs[i].set_aspect('auto')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 単語の配列
    words = 'I love natural language processing'

    # 例として、ランダムな自己注意の重みを生成
    num_heads = 2
    seq_length = len(words.split())
    attention_weights = torch.rand(num_heads, seq_length, seq_length)

    # 重みを可視化
    visualize_attention_weights(attention_weights, words)
