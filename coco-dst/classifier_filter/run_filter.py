"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import csv
import logging
import random
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup
try:
    from modeling import BertForMultiLabelSequenceClassification
except:
    from classifier_filter.modeling import BertForMultiLabelSequenceClassification
from tensorboardX import SummaryWriter
import json
from tqdm import tqdm
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DSTProcessor(DataProcessor):
    def __init__(self,root):

        self.D = [[], [], []]
        self.D[0] = self.load_data(os.path.join(root, 'train_dials.json'))
        self.D[1] = self.load_data(os.path.join(root, 'dev_dials.json'))
        self.D[2] = self.load_data(os.path.join(root, 'test_dials.json'))

    def load_data(self, path):
        multi_label_data = []
        labels = self.get_labels()
        with open(path) as f:
            data = json.load(f)
            for dial in data:
                dialog_history = ""
                for idx, turn in enumerate(dial["dialogue"]):
                    label_list = []
                    turn_domain = turn["domain"]
                    text_a = dialog_history
                    text_b =  turn["system_transcript"]+" "+ turn["transcript"]
                    flag = False
                    for label in turn["turn_label"]:
                        if(label[0] not in labels):
                            flag = True
                            break
                        label_list.append(label[0])

                    dialog_history = dialog_history+" "+turn["system_transcript"]+" "+ turn["transcript"]
                    dialog_history = dialog_history.strip()
                    if(not flag):
                        multi_label_data.append([text_a.strip(),text_b.strip(),label_list])
                
        return multi_label_data
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["attraction-area",
                "attraction-name",
                "attraction-type",
                "hotel-area",
                "hotel-book day",
                "hotel-book people",
                "hotel-book stay",
                "hotel-internet",
                "hotel-name",
                "hotel-parking",
                "hotel-pricerange",
                "hotel-stars",
                "hotel-type",
                "restaurant-area",
                "restaurant-book day",
                "restaurant-book people",
                "restaurant-book time",
                "restaurant-food",
                "restaurant-name",
                "restaurant-pricerange",
                "taxi-arriveby",
                "taxi-departure",
                "taxi-destination",
                "taxi-leaveat",
                "train-arriveby",
                "train-book people",
                "train-day",
                "train-departure",
                "train-destination",
                "train-leaveat"]
                
    def _create_examples(self,data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in data:
            text_a = d[0]
            text_b = d[1] 
            label = d[2]
            examples.append(InputExample(text_a=text_a,text_b = text_b,label=label))
            
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if example.text_c:
            tokens_c = tokenizer.tokenize(example.text_c)

        if tokens_c:
            truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
            tokens_b = tokens_b + ["[SEP]"] + tokens_c
        elif tokens_b:
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
                
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = len(label_map)*[0]
        for label in example.label:
            label_id[label_map[label]] = 1
        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    print('#features', len(features))
    return features


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop(0)


def truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop(0)
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop(0)
        else:
            tokens_c.pop(0)       


def metric(probs, labels,thresh):
    batch_size = probs.size(0)
    preds = (probs>thresh)
    preds = preds.cpu().numpy()
    labels = labels.byte().cpu().numpy()
    acc_list,precision_list,recall_list = [], [], []
    precision_numerator,precision_denominator , recall_numerator, recall_denominator = 0, 0, 0, 0
    for idx in range(batch_size):
        pred = preds[idx]
        label = labels[idx]
        acc = np.mean(pred==label)

        if(sum(pred) >0):
            precision = sum(label[pred==1]==1)/sum(pred)
        else:
            precision = 1
        if(sum(label) > 0):
            recall = sum(pred[label==1]==1)/sum(label)
        else:
            recall = 1
        precision_numerator += sum(label[pred==1]==1)
        precision_denominator += sum(pred)
        recall_numerator += sum(pred[label==1]==1)
        recall_denominator += sum(label)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
    return sum(acc_list),sum(precision_list),sum(recall_list), precision_numerator, precision_denominator, recall_numerator, recall_denominator

def acc_pred(probs,labels,label_list,thresh):

    batch_size = probs.size(0)
    preds = (probs>thresh)
    preds = preds.cpu().numpy()
    labels = labels.byte().cpu().numpy()
    prediction_list = []
    target_list = []
    for idx in range(batch_size):
        prediction_list.append([])
        target_list.append([])
        pred = preds[idx]
        label = labels[idx]
        for idx,each_pred in enumerate(pred):
            if(each_pred):
                prediction_list[-1].append(label_list[idx])

        for idx,each_label in enumerate(label):
            if(each_label):
                target_list[-1].append(label_list[idx])

    return prediction_list,target_list

def save_checkpoint(state, is_best, epoch ,output_dir):
    torch.save(state,os.path.join(output_dir,str(epoch)+"_model.pt"))
    if is_best:
        torch.save(state, os.path.join(output_dir,"best_model.pt"))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def convert_examples_to_tensor(examples,label_list,max_seq_length,tokenizer):
    
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    for f in features:
        input_ids.append(f.input_ids)
        input_mask.append(f.input_mask)
        segment_ids.append(f.segment_ids)
        label_id.append([f.label_id])                

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.float32)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    return data
    
    
def evaluate(model,device,eval_batch_size,eval_data,thresh = 0.5):
    
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    eval_accuracy, eval_precision ,eval_recall  = 0, 0, 0
    precision_numerator, precision_denominator, recall_numerator, recall_denominator = 0, 0, 0, 0
    nb_eval_examples = 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)
        # import pdb;
        # pdb.set_trace()
        probs = logits.sigmoid()
        tmp_acc, tmp_precision ,tmp_recall, tmp_precision_numerator, tmp_precision_denominator, tmp_recall_numerator, tmp_recall_denominator  = metric(probs, torch.squeeze(label_ids),thresh)
        eval_accuracy += tmp_acc
        eval_precision += tmp_precision
        eval_recall += tmp_recall
        nb_eval_examples += input_ids.size(0)
        precision_numerator += tmp_precision_numerator
        precision_denominator += tmp_precision_denominator
        recall_numerator += tmp_recall_numerator
        recall_denominator += tmp_recall_denominator

    eval_accuracy = eval_accuracy / nb_eval_examples
    macro_precision = eval_precision / nb_eval_examples
    macro_recall = eval_recall / nb_eval_examples
    micro_precision = precision_numerator/precision_denominator
    micro_recall = recall_numerator/recall_denominator
    return (eval_accuracy,macro_precision,macro_recall,micro_precision,micro_recall)

def eval_write(model,device,eval_batch_size,eval_data,label_list,thresh = 0.5):
    
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    prediction_list = []
    target_list = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

        probs = logits.sigmoid()
        batch_prediction_list,batch_target_list = acc_pred(probs, torch.squeeze(label_ids),label_list,thresh)
        prediction_list += batch_prediction_list
        target_list += batch_target_list
    
    return (prediction_list,target_list)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="max gradient norm")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
        
    args = parser.parse_args()
    
    print(args)
    writer = SummaryWriter(args.output_dir)

    processors = {
        "dst": DSTProcessor,
    }
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print(n_gpu, " are used during training!")
        
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](args.data_dir)
    label_list = processor.get_labels()

    bert_config = BertConfig.from_pretrained("bert-base-uncased",num_labels=len(processor.get_labels()))

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_examples = None
    num_train_steps = None
    
    
    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased",config = bert_config)
    

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.do_train:
        print('Loading training/dev data')
        train_examples = processor.get_train_examples(args.data_dir)
        train_data = convert_examples_to_tensor(train_examples, label_list, args.max_seq_length, tokenizer)
        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_data = convert_examples_to_tensor(dev_examples, label_list, args.max_seq_length, tokenizer)
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
            ]

        num_warmup_steps = int(num_train_steps*args.warmup_proportion)
        optimizer = AdamW(optimizer_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps= num_warmup_steps, num_training_steps=num_train_steps
        )
        
    if args.do_eval:
        print('Loading dev data')
        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_data = convert_examples_to_tensor(dev_examples, label_list, args.max_seq_length, tokenizer)
        print('Loading test data')
        test_examples = processor.get_test_examples(args.data_dir)
        test_data = convert_examples_to_tensor(test_examples, label_list, args.max_seq_length, tokenizer)

    global_step = 0
    if args.do_train:

        train_sampler = RandomSampler(train_data)    
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        best_macro_recall = - math.inf
        start = time.time()
        batch_num = len(train_dataloader)
        n_iters = args.num_train_epochs*batch_num
        print('Start training')
        print_every = n_iters//100 # Print time information every 1%
        for epoch in range(int(args.num_train_epochs)):
            iter_th = batch_num*epoch
            tr_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # import pdb;
                # pdb.set_trace()
                loss = model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask, labels = label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                iter_th += 1
                if (iter_th+1) % (print_every+1) == 0:
                    print('%s (%d %d%%)' % (timeSince(start, iter_th / n_iters),
                                         iter_th, iter_th / n_iters * 100))
                    
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()    # We have accumulated enought gradients
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                    writer.add_scalar('training/train_loss',tr_loss,global_step)
                    tr_loss = 0
            
            eval_accuracy,macro_precision,macro_recall,micro_precision,micro_recall = evaluate(model,device,args.eval_batch_size,dev_data)
            writer.add_scalar('training/dev_acc', eval_accuracy, epoch+1)
            writer.add_scalar('training/dev_macro_precision', macro_precision, epoch+1)
            writer.add_scalar('training/dev_macro_recall', macro_recall, epoch+1)
            writer.add_scalar('training/dev_micro_precision', micro_precision, epoch+1)
            writer.add_scalar('training/dev_micro_recall', micro_recall, epoch+1)
            if(macro_recall > best_macro_recall):
                best_macro_recall = macro_recall
                is_best = True
            else:
                is_best = False
                
            print("***** Eval on dev set *****")
            print("Current eval_acc = %.4f" % (eval_accuracy))
            print("Current macro_precision = %.4f" % (macro_precision))
            print("Current macro_recall = %.4f" % (macro_recall))
            print("Current micro_precision = %.4f" % (micro_precision))
            print("Current micro_recall = %.4f" % (micro_recall))
            try:
                model_dict = model.module.state_dict()
            except AttributeError:
                model_dict = model.state_dict()
            save_checkpoint(model_dict,is_best,epoch+1,args.output_dir)
    
        
    if(args.do_eval):
        
        model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased",config = bert_config)
        model.load_state_dict(torch.load(os.path.join(args.output_dir,"best_model.pt"), map_location='cpu'))
        model.to(device)
        if n_gpu > 1:
           model = torch.nn.DataParallel(model)
        eval_accuracy,macro_precision,macro_recall,micro_precision,micro_recall = evaluate(model,device,args.eval_batch_size,dev_data)
        print("***** Results of Best saved model*****")
        print("***** Eval on dev set *****")
        print("Current eval_acc = %.4f" % (eval_accuracy))
        print("Current macro_precision = %.4f" % (macro_precision))
        print("Current macro_recall = %.4f" % (macro_recall))
        print("Current micro_precision = %.4f" % (micro_precision))
        print("Current micro_recall = %.4f" % (micro_recall))
        eval_accuracy,macro_precision,macro_recall,micro_precision,micro_recall = evaluate(model,device,args.eval_batch_size,test_data)
        prediction_list, target_list = eval_write(model,device,args.eval_batch_size,test_data,label_list)
        print("***** Eval on test set *****")
        print("Current eval_acc = %.4f" % (eval_accuracy))
        print("Current macro_precision = %.4f" % (macro_precision))
        print("Current macro_recall = %.4f" % (macro_recall))
        print("Current micro_precision = %.4f" % (micro_precision))
        print("Current micro_recall = %.4f" % (micro_recall))
        result = []
        for (idx, example) in enumerate(test_examples):
            text = example.text_a
            if(example.text_b):
                text += (" "+example.text_b)
            if(example.text_c):
                text += (" "+example.text_c)
            result.append({"conetext:":text,
                            "pred":prediction_list[idx],
                            "target":target_list[idx]})
        with open(os.path.join(args.output_dir,"pred_result.json"),"w") as f:
            json.dump(result, f, indent=4)
        

if __name__ == "__main__":
    main()
 