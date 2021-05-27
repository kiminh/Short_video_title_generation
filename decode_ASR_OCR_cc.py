# coding=utf-8
# The MIT License (MIT)

# Copyright (c) Microsoft Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import argparse
import math
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from modeling_new_2_1_cc import UnilmForSeq2SeqDecode, UnilmConfig
# from transformers import (UnilmTokenizer, WhitespaceTokenizer,
#                           UnilmForSeq2SeqDecode, AdamW, UnilmConfig)


import utils_seq2seq_ASR_OCR
from collections import Counter


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (UnilmConfig,)), ())
MODEL_CLASSES = {
    'unilm': (UnilmConfig, UnilmForSeq2SeqDecode, UnilmTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def sents_index_locate(sent, str):
    indexs = []
    for i in range(len(sent)):
        if sent[i] == str:
            indexs.append(i)
    return indexs


def find_vid_idx(vid_att):
    sorted_idx = np.argsort(vid_att, axis=1).numpy()

    idx_weight = {}
    # give each index a weight (0-24) according to their 先后顺序 in each row and add,
    # then find the most important indexes with the biggest weights
    for i in range(sorted_idx.shape[0]):
        for j in range(sorted_idx.shape[1]):
            idx_weight[str(sorted_idx[i][j])] = 0

    for i in range(sorted_idx.shape[0]):
        for j in range(sorted_idx.shape[1]):
            idx_weight[str(sorted_idx[i][j])] += j

    idx_weight = Counter(idx_weight)

    return idx_weight


def refine(input, output, seq_id, beam_id, atten, max_len_tgt, vid_len, keep_num):

    end_sent1 = sents_index_locate(input, '。')
    end_sent2 = sents_index_locate(input, '！')
    end_sent3 = sents_index_locate(input, '？')
    end_sent = end_sent1 + end_sent2 + end_sent3
    end_sent = sorted(end_sent)
    input_len = len(input) + 2 + vid_len
    output_len = len(output.split(' ')) + 1
    if output_len > max_len_tgt:
        output_len = max_len_tgt

    all_att = tuple(
        [torch.zeros((1, 12, input_len + max_len_tgt, input_len + max_len_tgt)) for i in range(12)])
    for j in range(output_len):
        for k in range(12):
            if j == 0:
                a = atten[0][k][0].to('cpu')
                all_att[k][:, :, :a.size(2), :a.size(3)] = a[:, :, :, :]
            else:
                token_id = seq_id[0][j - 1].item()
                beams = beam_id[j - 1][0]
                pos = beams.index(token_id)
                attentions = atten[j]
                a = attentions[k][0][pos:pos + 1, :, -1:, :].to('cpu')
                all_att[k][:, :, a.size(3) - 1:a.size(3), :a.size(3)] = a[:, :, :, :]
    tokens = ['CLS'] + input + ['frame'] * vid_len + ['SEP'] + output.split(' ') + ['SEP']
    all_att = tuple([item[:, :, :len(tokens), :len(tokens)] for item in all_att])
    all_att_average = ()
    for i in range(len(all_att)):
        att = all_att[i]
        all_att_average += (torch.mean(att, dim=1),)
    att_src2gen = all_att_average[-1][0][input_len:-1, 1:len(input) + 1]
    att_vid2gen = all_att_average[-1][0][input_len:-1, len(input) + 1:input_len - 1]
    max_index = np.argmax(att_src2gen, axis=1)
    max_index = max_index.numpy().tolist()
    sent_ids = []
    for ind in max_index:
        sent_id = 0
        for end in end_sent:
            if ind < end:
                sent_ids.append(sent_id)
                break
            sent_id += 1
    sent_ids = [s[0] for s in Counter(sent_ids).most_common(keep_num)] #find 出现频次最大的三个

    vid_weight = find_vid_idx(att_vid2gen)
    vid_inds = [int(s[0]) for s in Counter(vid_weight).most_common(keep_num)]
    # max_index = np.argmax(att_vid2gen, axis=1)
    # max_index = max_index.numpy().tolist()
    # vid_inds = [s[0] for s in Counter(max_index).most_common(keep_num)]
    return sent_ids, sorted(vid_inds)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default='unilm', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='checkpoints/torch_unilm_model/', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--model_recover_path", default='output_2w_r0.2/model.20.bin', type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--config_name", default='', type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default='', type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--input_file", type=str, default='data/train_se_2w_r0.2.txt', help="Input file")
    parser.add_argument("--vid_file", type=str, default='data/train_vid_2w_r0.2.txt', help="vid file")
    parser.add_argument("--vid_path", type=str, default='data/train_2w/', help="vid path")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, default='output_2w_r0.2/train_.json', help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--max_tgt_length', type=int, default=50,
                        help="maximum length of target sequence")
    parser.add_argument('--keep_weight_num', type=int, default=3,
                        help="maximum number of refined labels")

    args = parser.parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, max_position_embeddings=args.max_seq_length)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    bi_uni_pipeline = []
    bi_uni_pipeline.append(utils_seq2seq_ASR_OCR.Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
                                                                  args.max_seq_length, max_tgt_length=args.max_tgt_length))

    # Prepare model
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_recover_path)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = model_class.from_pretrained(args.model_name_or_path, state_dict=model_recover, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size, length_penalty=args.length_penalty,
                                            eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len)
        del model_recover

        model.to(device)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length

        with open(args.input_file, encoding="utf-8") as fin:
            if 'train' not in args.input_file:
                input_lines = [x.strip() for x in fin.readlines()] #'{\'src_text\': \'有情调的人生就是这么精致和——繁琐！（转） \\u200b\\u200b\\u200b\', \'tgt_text\': \'喝什么酒用什么杯\'}'
            else:
                input_lines = [eval(x)['src_text'].strip() for x in fin.readlines()]
            if args.subset > 0:
                logger.info("Decoding subset: %d", args.subset)
                input_lines = input_lines[:args.subset]

        with open(args.vid_file, 'r') as ff:
            vid_lists = [line.split('\n')[0] for line in ff]
        vid_paths = [args.vid_path + ind for ind in vid_lists]
        vid_data = [np.load(vid_path) for vid_path in vid_paths]
        if args.subset > 0:
            vid_data = vid_data[:args.subset]

        data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
        input_lines = [data_tokenizer.tokenize(
            x)[:max_src_length] for x in input_lines] # ['{', '"', 'sr', '##c', '_', 'text', '"', ':', '"', '2020', '年', '去', '世', '的', '明', '星', '最', '后', '就', '会', '让', '人', '惋', '惜', '。', '"', ',', '"', 't', '##gt', '_', 'text', '"', ':', '"', '2020', '去', '世', '的', '明', '星', '最', '后', '一', '位', '让', '人', '惋', '惜', '"', '}'] for tencent
        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        sorted_index = [line[0] for line in input_lines]
        vid_data_sorted = [vid_data[ii] for ii in sorted_index]

        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        batch_ids = []
        # inputs = []
        # outputs = []
        # seq_ids = []
        # beam_ids = []
        # attens = []
        refined_ind_src = []
        refined_ind_vid = []

        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]

                _vid = vid_data_sorted[next_i:next_i + args.batch_size]
                _vid = tuple(_vid)
                vid_fea = torch.tensor(_vid, dtype=torch.float)
                # vid_fea = vid_fea.to(device)

                buf_id = [x[0] for x in _chunk]
                #batch_ids.append(buf_id)
                buf = [x[1] for x in _chunk]
                if buf[0] == []:
                    buf[0] = ['nan']
                batch_ids.append(buf_id)
                # inputs.append(buf)
                next_i += args.batch_size
                max_a_len = max([len(x) for x in buf])
                instances = []
                idx = 0
                for instance in [(x, max_a_len) for x in buf]:
                    vid = vid_fea[idx]
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance, vid))
                    idx = idx + 1
                with torch.no_grad():
                    batch = utils_seq2seq_ASR_OCR.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, vid, vid_start = batch
                    traces, all_attentions, step_ids = model(vid, vid_start, input_ids, token_type_ids,
                                   position_ids, input_mask, output_attentions=True)

                    # for i in range(len(all_attentions)):
                    #     for j in range(len(all_attentions[i])):
                    #         all_attentions[i][j][0] = all_attentions[i][j][0].to('cpu')
                    # seq_ids.append(traces['pred_seq'])
                    # beam_ids.append(step_ids)
                    # attens.append(all_attentions)
                    seq_id = traces['pred_seq']
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        output_lines[buf_id[i]] = output_sequence
                        # outputs.append(output_sequence)
                        if args.need_score_traces:
                            score_trace_list[buf_id[i]] = {
                                'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}


                #if buf[0] == '':
                #    buf[0] = ' '
                #    print(buf[0])

                sent_ids, vid_inds = refine(buf[0], output_sequence, seq_id, step_ids, all_attentions, args.max_tgt_length, vid.size(1), args.keep_weight_num)
                # input = buf[0]
                # end_sent1 = sents_index_locate(input, '。')
                # end_sent2 = sents_index_locate(input, '！')
                # end_sent3 = sents_index_locate(input, '？')
                # end_sent = end_sent1 + end_sent2 + end_sent3
                # end_sent = sorted(end_sent)
                #
                # input_len = len(input) + 2 + 25
                #
                # output = output_sequence
                # output_len = len(output.split(' ')) + 1
                # seq_id = traces['pred_seq'].to('cpu')
                # beam_id = step_ids
                # atten = all_attentions
                #
                # all_att = tuple(
                #     [torch.zeros((1, 12, input_len + args.max_len_tgt, input_len + args.max_len_tgt)) for i in range(12)])
                # for j in range(output_len):
                #     for k in range(12):
                #         if j == 0:
                #             a = atten[0][k][0].to('cpu')
                #             all_att[k][:, :, :a.size(2), :a.size(3)] = a[:, :, :, :]
                #         else:
                #             token_id = seq_id[0][j - 1].item()
                #             beams = beam_id[j - 1][0]
                #             pos = beams.index(token_id)
                #             attentions = atten[j]
                #             a = attentions[k][0][pos:pos + 1, :, -1:, :].to('cpu')
                #             all_att[k][:, :, a.size(3) - 1:a.size(3), :a.size(3)] = a[:, :, :, :]
                # tokens = ['CLS'] + input + ['frame'] * 25 + ['SEP'] + output.split(' ') + ['SEP']
                # all_att = tuple([item[:, :, :len(tokens), :len(tokens)] for item in all_att])
                # all_att_average = ()
                # for i in range(len(all_att)):
                #     att = all_att[i]
                #     all_att_average += (torch.mean(att, dim=1),)
                # att_src2gen = all_att_average[-1][0][input_len:-1, 1:len(input) + 1]
                # att_vid2gen = all_att_average[-1][0][input_len:-1, len(input) + 1:input_len-1]
                # max_index = np.argmax(att_src2gen, axis=1)
                # sent_ids = []
                # for ind in max_index:
                #     sent_id = 0
                #     for end in end_sent:
                #         if ind < end:
                #             sent_ids.append(sent_id)
                #             break
                #         sent_id += 1
                # from collections import Counter
                # sent_ids = [s[0] for s in Counter(sent_ids).most_common(2)]
                # max_index = np.argmax(att_vid2gen, axis=1)
                # vid_ind = [s[0] for s in Counter(max_index).most_common(2)]
                refined_ind_src.append(sent_ids)
                refined_ind_vid.append(vid_inds)
                del all_attentions
                pbar.update(1)
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path+'.'+args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")

        if args.need_score_traces:
            with open(fn_out + ".trace.pickle", "wb") as fout_trace:
                pickle.dump(
                    {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
                for x in score_trace_list:
                    pickle.dump(x, fout_trace)

        # pickle.dump((batch_ids, inputs, outputs, seq_ids, beam_ids, attens), open('output_2w_r0.2/att_visualize.pkl', 'wb'))
        if 'train' in args.input_file:

            pickle.dump((batch_ids, refined_ind_vid, refined_ind_src),
                    open(fn_out.split('/')[0] + '/train_refined_ids.pkl', 'wb'))
        if 'val' in args.input_file:
            pickle.dump((batch_ids, refined_ind_vid, refined_ind_src), open(fn_out.split('/')[0] + '/val_refined_ids.pkl', 'wb'))
        if 'test' in args.input_file:
            pickle.dump((batch_ids, refined_ind_vid, refined_ind_src), open(fn_out.split('/')[0] + '/test_refined_ids.pkl', 'wb'))

if __name__ == "__main__":
    main()
