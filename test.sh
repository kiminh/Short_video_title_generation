#!/bin/bash

python decode_ASR_OCR.py  --model_recover_path checkpoints/model.14.bin --max_seq_length 512 --input_file data/test_raw.txt --input_label data/test_label.txt --vid_file data/test_vid.txt --vid_path data/train_2w/ --output_file output/predict.json --do_lower_case --batch_size 16 --beam_size 5 --max_tgt_length 50
