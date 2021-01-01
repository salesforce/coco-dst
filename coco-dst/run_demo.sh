# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

export CUDA_VISIBLE_DEVICES=0
eval_data_file="../multiwoz/MultiWOZ_2.1/test_dials.json"
model_name_or_path="./coco_model/checkpoint-12000"
python run_demo.py \
    --eval_data_file=${eval_data_file} \
    --model_type=t5 \
    --model_name_or_path=${model_name_or_path} \
    --num_beams=5 \
    --k=0 \
    --p=1 \
    --length=100 \
    --num_return_sequences=5 \
    --temperature=1 \
    --seed=42 \
    --repetition_penalty=1.0 \
    --shuffle_turn_label \

