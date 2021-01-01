# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
class ArgsParser(object):
     
     def __init__(self):

          parser = argparse.ArgumentParser()
          parser.add_argument("--train_data_file", default=None, type=str, required=True,
                              help="Training data")
          parser.add_argument("--eval_data_file", default=None, type=str, required=True,
                              help="Dev data file")
          parser.add_argument("--model_type", default=None, type=str, required=True,
                              help="Model type selected in the list: [t5] ")
          parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                              help="Path to pre-trained model or shortcut name selected in the list:")
          parser.add_argument("--output_dir", default=None, type=str, required=True,
                              help="The output directory where the model checkpoints and predictions will be written.")
          parser.add_argument("--config_name", default=None, type=str,
                              help="Pretrained config name or path if not the same as model_name")
          parser.add_argument("--tokenizer_name", default=None, type=str,
                              help="Pretrained tokenizer name or path if not the same as model_name")
          parser.add_argument("--cache_dir", default=None, type=str,
                              help="Where do you want to store the pre-trained models downloaded from s3")
          parser.add_argument("--max_src_len", default=512, type=int,
                              help="The maximum total encoder sequence length."
                                   "Longer than this will be truncated, and sequences shorter than this will be padded.")
          parser.add_argument("--max_tgt_len", default=512, type=int,
                              help="The maximum total decoder sequence length."
                                   "Longer than this will be truncated, and sequences shorter than this will be padded.")
          parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                              help="Batch size per GPU/CPU for training.")
          parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                              help="Batch size per GPU/CPU for training.")
          parser.add_argument("--save_total_limit", default=-1, type=int,
                              help="maximum of checkpoint to be saved")
          parser.add_argument("--learning_rate", default=5e-5, type=float,
                              help="The initial learning rate for Adam.")
          parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                              help="Number of updates steps to accumulate before performing a backward/update pass.")
          parser.add_argument("--weight_decay", default=0.00, type=float,
                              help="Weight decay if we apply some.")
          parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                              help="Epsilon for Adam optimizer.")
          parser.add_argument("--max_grad_norm", default=1.0, type=float,
                              help="Max gradient norm.")
          parser.add_argument("--num_train_steps", default=-1, type=int,
                              help="set total number of training steps to perform")
          parser.add_argument("--num_train_epochs", default=10, type=int,
                              help="set total number of training epochs to perform (--num_training_steps has higher priority)")
          parser.add_argument("--num_warmup_steps", default=0, type=int,
                              help="Linear warmup over warmup_steps.")
          parser.add_argument('--logging_steps', type=int, default=500,
                              help="Log every X updates steps.")
          parser.add_argument('--save_steps', type=int, default=1500,
                              help="Save checkpoint every X updates steps.")
          parser.add_argument("--no_cuda", action='store_true',
                              help="Whether not to use CUDA when available")
          parser.add_argument("--shuffle_turn_label", action='store_true',
                              help="do training")
          parser.add_argument("--do_train", action='store_true',
                              help="do training")
          parser.add_argument("--do_eval", action='store_true',
                              help="do eval")
          parser.add_argument("--evaluate_during_training", action='store_true',
                              help="evaluate_during_training")
          parser.add_argument("--eval_all_checkpoints", action='store_true',
                              help="evaluate_during_training")
          parser.add_argument("--should_continue", action='store_true',
                              help="If we continue training from a checkpoint")
          parser.add_argument('--seed', type=int, default=42,
                              help="random seed for initialization")
          parser.add_argument("--local_rank", type=int, default=-1,
                              help="local_rank for distributed training on gpus")
          parser.add_argument('--fp16', action='store_true',
                              help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
          parser.add_argument('--fp16_opt_level', type=str, default='O1',
                              help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                   "See details at https://nvidia.github.io/apex/amp.html")
          parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
          parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

          self.parser = parser

     def parse(self):
          args = self.parser.parse_args()
          return args
