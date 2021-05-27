#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python decode_ASR_OCR_cc.py --model_recover_path checkpoints/model.14.bin --max_seq_length 512 --input_file data/train_raw.txt --vid_file data/train_vid.txt --vid_path data/train_2w/ --output_file output/train.json --do_lower_case --batch_size 1 --beam_size 5 --max_tgt_length 50 --keep_weight_num 3
