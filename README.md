# Short_video_title_generation
VTG for video title generation and cover selection
The code is deployed on Ubuntu 20.04, and the required environment is listed in requirement.yml.

The TSVTG dataset is available on https://drive.google.com/drive/folders/1wO1UBSXRzFEhO6V1sgKmg6d6-FckXMNv?usp=sharing.

Experiment steps:
1. To extract video frames and frame features:
nohup python frames_extraction.py --video_path 'data/video/smalldata_asr_rest' --frame_path 'data/frame/smalldata_asr_rest' > log2.log 2>&1 &

--video_path: the path your videos are saved
--frame_path: the path the extracted frames are saved

nohup python frame_feats_extraction.py --frame_path 'data/frame/smalldata_asr_rest' --feat_path 'data/feats/smalldata_asr_rest' > log3.log 2>&1 &

--feat_path: the path the extracted frame features are saved

2. Train/Fine-tune the MSVTG model and save the checkpoints in args.output_file
bash train.sh

3. Test the checkpoints using the validation set
bash val.sh

Select the best model with the highest mean score to evaluate the test data
bash test.sh

4. Decode train data and generate the selected sentence/frame id and generated title for each sample,
just a few revision to the input file path when generating the refine ids for validation/test data, but please note we never use the label of test data in the whole experiments expect in test.sh.
bash test_train.sh

5. Generate the refined text and frame features without sample level refinement

python refine_txt_no_samples_reduce.py --refined_file_path output/test_ --generated_title output/test_.json --model_input data/test_raw.txt --model_input_label data/test_label.txt --model_input_ids data/test_vid.txt --vid_path data/train_2w --output_file1 data/test_refined

Or generate the refined text and frame features with sample level refinement (only for train data):

python refine_txt_samples_reduce.py --refined_file_path output/train_ --generated_title output/train.json --model_input data/train_0.2_raw.txt --model_input_label data/train_raw.txt --model_input_ids data/train_vid.txt --vid_path data/train_2w --output_file1 data/train_refined

6. After obtaining the refined text and frame features, repeat from 2 until you get the ideal results you expect.
Note: our best model is saved in the checkpoints folder for reference or future use

