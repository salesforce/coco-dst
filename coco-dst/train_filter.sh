# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

export CUDA_VISIBLE_DEVICES=0
save_dir='./classifier_filter/filter'
data_dir="../multiwoz/MultiWOZ_2.2/MultiWOZ2.1Format"
if [ ! -d "${save_dir}" ]; then
  mkdir -p ${save_dir}
fi
python ./classifier_filter/run_filter.py --task_name DST --do_train --do_lower_case --data_dir ${data_dir} --max_seq_length 512 --train_batch_size 12 --learning_rate 2e-5 --num_train_epochs 5 --output_dir ${save_dir} --gradient_accumulation_steps 1 --seed 0
