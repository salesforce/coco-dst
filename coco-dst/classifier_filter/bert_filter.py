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
import numpy as np
import torch
from classifier_filter.run_filter import *
from transformers import BertConfig, BertTokenizer
from classifier_filter.modeling import BertForMultiLabelSequenceClassification
import json

class DataProcessor():
    def __init__(self,path):

        self.data = self.load_data(path)

    def load_data(self,path):
        multi_label_data = {}
        with open(path) as f:
            data = json.load(f)
            for dial in data:
                dialog_history = ""
                for idx, turn in enumerate(dial["dialogue"]):
                    label_list = []
                    turn_domain = turn["domain"]
                    text_a = dialog_history
                    text_b =  turn["system_transcript"]
                    dialog_history = dialog_history+" "+turn["system_transcript"]+" "+ turn["transcript"]
                    dialog_history = dialog_history.strip()
                    multi_label_data[dial["dialogue_idx"]+str(idx)] = {"text_a":text_a,
                                                                       "text_b":text_b,
                                                                       "label_list":label_list}

        return multi_label_data

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
                
    def create_examples(self,dialogue_idx,turn_id,user_utters,turn_label):

        examples = []
        meta_info = self.data[dialogue_idx+str(turn_id)]
        for user_utter in user_utters:
            text_a = meta_info["text_a"]
            text_b = meta_info["text_b"]+" "+user_utter
            labels = []
            for label in turn_label:
                labels.append(label[0])
            # print("text_a: ",text_a.strip())
            # print("text_b: ",text_b.strip())
            # print("*************************")
            examples.append(InputExample(text_a=text_a.strip(),text_b = text_b.strip(),label=labels))
            
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

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

    return features


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

    data = (all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    return data

class BERTFilter(object):

    def __init__(self,data_file):

        self.processor = DataProcessor(data_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_list = self.processor.get_labels()
        bert_config = BertConfig.from_pretrained("bert-base-uncased",num_labels=len(self.label_list))
        self.max_seq_length = 512
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased",config = bert_config)
#         import pdb;
#         pdb.set_trace();
#         import sys
        self.model.load_state_dict(torch.load("./classifier_filter/filter/best_model.pt", map_location='cpu'))
        self.model.to(self.device)

    def query_filter(self,dialogue_idx,turn_id,user_utters,turn_label,thresh):

        examples = self.processor.create_examples(dialogue_idx,turn_id,user_utters,turn_label)
        data = convert_examples_to_tensor(examples, self.label_list, self.max_seq_length, self.tokenizer)
        result = self.evaluation(data,thresh)
        # print(result)
        return result
    def evaluation(self,data,thresh):

        self.model.eval()
        prediction_list = []
        target_list = []
        input_ids, input_mask, segment_ids, label_ids  = data
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)
        label_ids = label_ids.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask)

        probs = logits.sigmoid()
        prediction_list,target_list = self.acc_pred(probs, label_ids.view(-1,len(self.label_list)),self.label_list,thresh)
        result = []
        for idx in range(len(prediction_list)):
            prediction_set = set(prediction_list[idx])
            target_set = set(target_list[idx])
            # print("pred: ",prediction_set)
            # print("target: ",target_set)
            # print("*************************")
            if(prediction_set.issubset(target_set)):
                result.append(True)
            else:
                result.append(False)
    
        return result

    def acc_pred(self,probs,labels,label_list,thresh):

        batch_size = probs.size(0)
        preds = (probs>thresh)
        preds = preds.cpu().numpy()
        labels = labels.byte().cpu().numpy()
        prediction_list = []
        target_list = []
        for idx in range(batch_size):
            pred = preds[idx]
            label = labels[idx]
            prediction_list.append([])
            target_list.append([])
            for idx,each_pred in enumerate(pred):
                if(each_pred):
                    prediction_list[-1].append(label_list[idx])

            for idx,each_label in enumerate(label):
                if(each_label):
                    target_list[-1].append(label_list[idx])

        return prediction_list,target_list

        

if __name__ == "__main__":
    classifier_filter = BERTFilter()
    while(True):
        dialogue_idx = "PMUL3688.json"
        turn_id = 4
        thresh=0.5
        user_utters =["that will work. i will need tickets for 3 people.", "that will work. thank you."]
        turn_label = [            
            [
                "train-book people",
                "3"
            ]
        ]
        flag = classifier_filter.query_filter(dialogue_idx,turn_id,user_utters,turn_label,thresh)
        import pdb;
        pdb.set_trace()
    
 