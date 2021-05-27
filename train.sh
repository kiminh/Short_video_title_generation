#!/bin/bash
#train 
python run_ASR_OCR.py --src_file 'train_raw.txt' --vid_file 'train_vid.txt' --vid_path data/train_2w/ --output_dir checkpoints/  --train_batch_size 16 --num_train_epochs 20 --num_workers 32 --logging_steps 296

#fine tune
python run_ASR_OCR.py --src_file 'train_raw.txt' --vid_file 'train_vid.txt' --vid_path data/train_2w/ --model_recover_path checkpoints/model.14.bin --optim_recover_path checkpoints/optim.14.bin --output_dir output/  --train_batch_size 16 --num_train_epochs 20 --num_workers 32 --logging_steps 296
