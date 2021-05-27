import pickle
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.switch_backend('agg')
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = ['AR PL UKai CN']


all_att_src, all_att_vid, tokens = pickle.load(open('output_raw16k_withvid_0.2/exmaple.pkl', 'rb'))


def showAttention(input_sentence, output_words, attentions, img_path):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(15, 5.1), dpi=200) ##for src
    # fig = plt.figure(figsize=(1.5, 5.4), dpi=200) ##for vid
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='Reds') #bone
    # fig.colorbar(cax)

    divider = make_axes_locatable(ax)
    # fig.colorbar(cax, cax=divider.append_axes("right", size="2%", pad=0.05))

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig(img_path)


input_tokens = tokens[:all_att_src.size(1)]
input_tokens = tokens[:50]
output_tokens = tokens[-(all_att_src.size(0)+1):-1]
showAttention(input_tokens, output_tokens, all_att_src[:, :50], 'output_raw16k_withvid_0.2/examplesrc.svg')

input_tokens = ['frame' + str(i) for i in range(1, all_att_vid.size(1) + 1)]
output_tokens = tokens[-(all_att_src.size(0)+1):-1]
showAttention(input_tokens, output_tokens, all_att_vid, 'output_raw16k_withvid_0.2/examplevid.svg')
