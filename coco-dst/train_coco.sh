# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

TRAIN_DATA_FILE=../multiwoz/MultiWOZ_2.2/MultiWOZ2.1Format/train_dials.json
EVAL_DATA_FILE=../multiwoz/MultiWOZ_2.2/MultiWOZ2.1Format/dev_dials.json
MODEL_NAME_OR_PATH=t5-small
OUTPUT_DIR=coco_model
export CUDA_VISIBLE_DEVICES=0
python train_coco.py \
  --do_train \
  --evaluate_during_training \
  --train_data_file ${TRAIN_DATA_FILE} \
  --eval_data_file ${EVAL_DATA_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --model_type t5 \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_src_len 100 \
  --max_tgt_len 100 \
  --per_gpu_train_batch_size  36 \
  --per_gpu_eval_batch_size 36 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --eval_all_checkpoints \
  --learning_rate 5e-5 \
  --num_warmup_steps 200 \
  --num_train_epochs 10 \
  --save_steps 1000 \
  --shuffle_turn_label
