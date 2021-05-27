#!/bin/bash

for i in $(seq 1 20)
do

	python decode_ASR_OCR.py  --model_recover_path checkpoints/model.$i.bin --max_seq_length 512 --input_file data/val_raw.txt --input_label data/val_label.txt --vid_file data/val_vid.txt --vid_path data/train_2w/ --output_file output/val_$i.json --do_lower_case --batch_size 16 --beam_size 5 --max_tgt_length 50

done
