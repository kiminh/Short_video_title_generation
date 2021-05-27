import pickle
import argparse
import logging
import json
import os
import numpy as np
from rouge import Rouge
import re
import jieba

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_json(file):
    data = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            e = json.loads(line)
            data.append(e)
    return data


def write_json(data, savefile):
    with open(savefile, 'w', encoding='utf8') as f:
        for i in data:
            f.write(json.dumps(i))
            f.write('\n')


def write_txt(src, tgt, savefile):
    with open(savefile, 'w', encoding='utf-8') as f:
        for i in range(len(src)):
            e = {}
            e['src_text'] = src[i]
            e['tgt_text'] = tgt[i]
            f.write(str(e))
            f.write('\n')

def write_val_txt(src, savefile):
    with open(savefile, 'w', encoding='utf-8') as f:
        for i in range(len(src)):
            f.write(src[i])
            f.write('\n')



def refine_video(vid_path, ind_vid_re, savefile):
    vid = np.load(vid_path)
    vid = vid[ind_vid_re, :]
    np.save(savefile, vid)


def sentence_split(sentences):
    new_text = []
    text = re.split('(。|！|\!|？)', sentences)
    if len(text) == 1:
        new_text.append(text[0])
    else:
        for i in range(int(len(text)/2)):
            sent = text[2*i] + text[2*i+1]
            new_text.append(sent)
        if len(text) % 2 != 0 and len(text[-1]) > 0:
            new_text.append(text[-1])
    return new_text


def compute_rouge(ref, hyp):
    rouge = Rouge()
    if hyp == '':
        hyp = ' '
    hyp = jieba.cut(hyp)
    hyp = ' '.join(hyp)

    ref = jieba.cut(ref)
    ref = ' '.join(ref)
    r2 = rouge.get_scores(hyp, ref)[0]['rouge-2']['f']

    return r2





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refined_file_path', default='output_raw16k_withvid_0.2/val_', type=str,
                        help='path of the refined ids for each train/val data')
    parser.add_argument('--generated_title', default='output_raw16k_withvid_0.2/val_.json', type=str,
                        help='path of the generated title for each train/val data')
    parser.add_argument('--model_input', default='data/val_0.2_raw.txt', type=str,
                        help='original data for input')
    parser.add_argument('--model_input_label', default='data/val_0.2_label.txt', type=str,
                        help='original label for input')
    parser.add_argument('--model_input_ids', default='data/val_0.2_se_vid.txt', type=str,
                        help='the id of the original data for input')
    parser.add_argument('--vid_path', default='data/train_2w', type=str,
                        help='the vid path for input')
    parser.add_argument('--subset', type=int, default=0,
                        help="deal with a subset of the input.")
    parser.add_argument('--output_file1', default='data/val_0.2_refined', type=str,
                        help='the refined vid input')
    args = parser.parse_args()


    if not os.path.exists(args.output_file1):
        os.mkdir(args.output_file1)

    with open(args.model_input, encoding='utf-8') as f:
        if 'train' in args.model_input:
            src_lines = [eval(x)['src_text'].strip() for x in f.readlines()]
        else:
            src_lines = [x.split('\n')[0] for x in f.readlines()]
    with open(args.model_input_label, encoding='utf-8') as f:
        if 'train' in args.model_input_label:
            tgt_lines = [eval(x)['tgt_text'].strip() for x in f.readlines()]
        else:
            tgt_lines = [x.split('\n')[0] for x in f.readlines()]

    with open(args.model_input_ids, encoding='utf-8') as f:
        ids = [x.split('.npy')[0] for x in f.readlines()]

    if args.subset > 0:
        src_lines = src_lines[:args.subset]
        tgt_lines = tgt_lines[:args.subset]
        ids = ids[:args.subset]

    batch_ids, refined_ind_vid, refined_ind_src = pickle.load(open(args.refined_file_path + 'refined_ids.pkl', 'rb'))
    with open(args.generated_title, 'r', encoding='utf-8') as f:
        gen = [''.join(line.split('\n')[0].split()) for line in f]


    for i in range(len(batch_ids)):
        print(batch_ids[i])
        id = ids[batch_ids[i][0]]
        generate = gen[batch_ids[i][0]]
        vid_path = os.path.join(args.vid_path, id + '.npy')
        output_file1 = os.path.join(args.output_file1, id + '.npy')
        ind_vid_re = refined_ind_vid[i]
        refine_video(vid_path, ind_vid_re, output_file1)
        summary = tgt_lines[batch_ids[i][0]]
        r2 = compute_rouge(summary, generate)
        src = src_lines[batch_ids[i][0]]
        src_list = sentence_split(src)
        ind_src_re = refined_ind_src[i] ### sorted(ind_src_re) maybe better

        src_re = [src_list[ind] for ind in ind_src_re]
        src_re = ' '.join(src_re)
        src_re = ''.join(src_re.split()) #refined src
        # summary = tgt_lines[batch_ids[i][0]]
        logger.info('\n The ground truth is {} \n The original src is {} \n '
                    'The refined src is {}'.format(summary, src, src_re))
        # src_lines[batch_ids[i][0]] = src_re # append the refined src to src
        src_lines[batch_ids[i][0]] = src + src_re

    if 'val' in args.model_input:
        # write_val_txt(src_lines, args.model_input.split('/')[0] + '/refined_' + args.model_input.split('/')[1])
        write_val_txt(src_lines, args.model_input.split('/')[0] + '/refined_append_' + args.model_input.split('/')[1])
    if 'test' in args.model_input:
        # write_val_txt(src_lines, args.model_input.split('/')[0] + '/refined_' + args.model_input.split('/')[1])
        write_val_txt(src_lines, args.model_input.split('/')[0] + '/refined_append_' + args.model_input.split('/')[1])
    if 'train' in args.model_input:
        # write_txt(src_lines, tgt_lines, args.model_input.split('/')[0] + '/refined_' + args.model_input.split('/')[1])
        write_txt(src_lines, tgt_lines, args.model_input.split('/')[0] + '/refined_append_' + args.model_input.split('/')[1])


if __name__ == "__main__":
    main()
