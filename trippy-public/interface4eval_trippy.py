# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import glob
import json
import math
import re

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from transformers import (AdamW, get_linear_schedule_with_warmup)

from modeling_bert_dst import (BertForDST)
from data_processors import PROCESSORS
from utils_dst import (convert_examples_to_features)
from tensorlistdataset import (TensorListDataset)
from interface_metric_bert_dst import *

logger = logging.getLogger(__name__)

MODEL_CHECKPOINT="../trippy-public/baseline/checkpoint-11810"
#MODEL_CHECKPOINT="../trippy-public/coco-vs_rare/checkpoint-9444"

MODEL_CLASSES = {
    'bert': (BertConfig, BertForDST, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

        
def to_list(tensor):
    return tensor.detach().cpu().tolist()


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append({k: v.to(device) for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)

def evaluate(args, model, tokenizer, processor, dialog_id, turn_id, new_usr_utter,new_turn_label,prefix=""):


    dataset, features = load_and_cache_examples(args, model, tokenizer, processor, dialog_id, turn_id, new_usr_utter,new_turn_label, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset) # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    with torch.no_grad():
        diag_state = {slot: torch.tensor([0 for _ in range(args.eval_batch_size)]).to(args.device) for slot in model.slot_list}
    for batch in eval_dataloader:
        model.eval()
        batch = batch_to_device(batch, args.device)

        # Reset dialog state if turn is first in the dialog.
        turn_itrs = [features[i.item()].guid.split('-')[2] for i in batch[9]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():
            inputs = {'input_ids':       batch[0],
                      'input_mask':      batch[1],
                      'segment_ids':     batch[2],
                      'start_pos':       batch[3],
                      'end_pos':         batch[4],
                      'inform_slot_id':  batch[5],
                      'refer_id':        batch[6],
                      'diag_state':      diag_state,
                      'class_label_id':  batch[8]}
            unique_ids = [features[i.item()].guid for i in batch[9]]
            values = [features[i.item()].values for i in batch[9]]
            input_ids_unmasked = [features[i.item()].input_ids_unmasked for i in batch[9]]
            inform = [features[i.item()].inform for i in batch[9]]
            outputs = model(**inputs)

            # Update dialog state for next turn.
            for slot in model.slot_list:
                updates = outputs[2][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

        results = eval_metric(model, inputs, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5])
        preds, ds = predict_and_format(model, tokenizer, inputs, outputs[2], outputs[3], outputs[4], outputs[5], unique_ids, input_ids_unmasked, values, inform, prefix, ds)
        all_results.append(results)
        all_preds.append(preds)

    all_preds = [item for sublist in all_preds for item in sublist]
    return all_preds


def eval_metric(model, features, total_loss, per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits):
    metric_dict = {}
    per_slot_correctness = {}
    for slot in model.slot_list:
        per_example_loss = per_slot_per_example_loss[slot]
        class_logits = per_slot_class_logits[slot]
        start_logits = per_slot_start_logits[slot]
        end_logits = per_slot_end_logits[slot]
        refer_logits = per_slot_refer_logits[slot]

        class_label_id = features['class_label_id'][slot]
        start_pos = features['start_pos'][slot]
        end_pos = features['end_pos'][slot]
        refer_id = features['refer_id'][slot]

        _, class_prediction = class_logits.max(1)
        class_correctness = torch.eq(class_prediction, class_label_id).float()
        class_accuracy = class_correctness.mean()

        # "is pointable" means whether class label is "copy_value",
        # i.e., that there is a span to be detected.
        token_is_pointable = torch.eq(class_label_id, model.class_types.index('copy_value')).float()
        _, start_prediction = start_logits.max(1)
        start_correctness = torch.eq(start_prediction, start_pos).float()
        _, end_prediction = end_logits.max(1)
        end_correctness = torch.eq(end_prediction, end_pos).float()
        token_correctness = start_correctness * end_correctness
        token_accuracy = (token_correctness * token_is_pointable).sum() / token_is_pointable.sum()
        # NaNs mean that none of the examples in this batch contain spans. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if math.isnan(token_accuracy):
            token_accuracy = torch.tensor(1.0, device=token_accuracy.device)

        token_is_referrable = torch.eq(class_label_id, model.class_types.index('refer') if 'refer' in model.class_types else -1).float()
        _, refer_prediction = refer_logits.max(1)
        refer_correctness = torch.eq(refer_prediction, refer_id).float()
        refer_accuracy = refer_correctness.sum() / token_is_referrable.sum()
        # NaNs mean that none of the examples in this batch contain referrals. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if math.isnan(refer_accuracy) or math.isinf(refer_accuracy):
            refer_accuracy = torch.tensor(1.0, device=refer_accuracy.device)
            
        total_correctness = class_correctness * (token_is_pointable * token_correctness + (1 - token_is_pointable)) * (token_is_referrable * refer_correctness + (1 - token_is_referrable))
        total_accuracy = total_correctness.mean()

        loss = per_example_loss.mean()
        metric_dict['eval_accuracy_class_%s' % slot] = class_accuracy
        metric_dict['eval_accuracy_token_%s' % slot] = token_accuracy
        metric_dict['eval_accuracy_refer_%s' % slot] = refer_accuracy
        metric_dict['eval_accuracy_%s' % slot] = total_accuracy
        metric_dict['eval_loss_%s' % slot] = loss
        per_slot_correctness[slot] = total_correctness

    goal_correctness = torch.stack([c for c in per_slot_correctness.values()], 1).prod(1)
    goal_accuracy = goal_correctness.mean()
    metric_dict['eval_accuracy_goal'] = goal_accuracy
    metric_dict['loss'] = total_loss
    return metric_dict


def predict_and_format(model, tokenizer, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits, ids, input_ids_unmasked, values, inform, prefix, ds):
    prediction_list = []
    dialog_state = ds
    for i in range(len(ids)):
        if int(ids[i].split("-")[2]) == 0:
            dialog_state = {slot: 'none' for slot in model.slot_list}

        prediction = {}
        prediction_addendum = {}
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            end_logits = per_slot_end_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]

            input_ids = features['input_ids'][i].tolist()
            class_label_id = int(features['class_label_id'][slot][i])
            start_pos = int(features['start_pos'][slot][i])
            end_pos = int(features['end_pos'][slot][i])
            refer_id = int(features['refer_id'][slot][i])
            
            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            prediction['guid'] = ids[i].split("-")
            prediction['class_prediction_%s' % slot] = class_prediction
            prediction['class_label_id_%s' % slot] = class_label_id
            prediction['start_prediction_%s' % slot] = start_prediction
            prediction['start_pos_%s' % slot] = start_pos
            prediction['end_prediction_%s' % slot] = end_prediction
            prediction['end_pos_%s' % slot] = end_pos
            prediction['refer_prediction_%s' % slot] = refer_prediction
            prediction['refer_id_%s' % slot] = refer_id
            prediction['input_ids_%s' % slot] = input_ids

            if class_prediction == model.class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == model.class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])
                dialog_state[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                dialog_state[slot] = re.sub("(^| )##", "", dialog_state[slot])
            elif 'true' in model.class_types and class_prediction == model.class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in model.class_types and class_prediction == model.class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == model.class_types.index('inform'):
                dialog_state[slot] = '§§' + inform[i][slot]
            # Referral case is handled below

            prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]
            prediction_addendum['slot_groundtruth_%s' % slot] = values[i][slot]

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in model.slot_list:
            class_logits = per_slot_class_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]
            
            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in model.class_types and class_prediction == model.class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                dialog_state[slot] = dialog_state[model.slot_list[refer_prediction - 1]]
                prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot] # Value update

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)
        
    return prediction_list, dialog_state


def load_and_cache_examples(args, model, tokenizer, processor, dialog_id, turn_id, new_usr_utter,new_turn_label, evaluate=False):

    # import pdb;
    # pdb.set_trace()

    logger.info("Creating features from dataset file at %s", args.data_dir)
    processor_args = {'append_history': args.append_history,
                      'use_history_labels': args.use_history_labels,
                      'swap_utterances': args.swap_utterances,
                      'label_value_repetitions': args.label_value_repetitions,
                      'delexicalize_sys_utts': args.delexicalize_sys_utts}

    if evaluate and args.predict_type == "test":
        examples = processor.get_example(processor_args, dialog_id, turn_id, new_usr_utter,new_turn_label)
    features = convert_examples_to_features(examples=examples,
                                            slot_list=model.slot_list,
                                            class_types=model.class_types,
                                            model_type=args.model_type,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            slot_value_dropout=(0.0 if evaluate else args.svd))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    f_start_pos = [f.start_pos for f in features]
    f_end_pos = [f.end_pos for f in features]
    f_inform_slot_ids = [f.inform_slot for f in features]
    f_refer_ids = [f.refer_id for f in features]
    f_diag_state = [f.diag_state for f in features]
    f_class_label_ids = [f.class_label_id for f in features]
    all_start_positions = {}
    all_end_positions = {}
    all_inform_slot_ids = {}
    all_refer_ids = {}
    all_diag_state = {}
    all_class_label_ids = {}
    for s in model.slot_list:
        all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
        all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos], dtype=torch.long)
        all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
        all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
        all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
        all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)
    dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_inform_slot_ids,
                                all_refer_ids,
                                all_diag_state,
                                all_class_label_ids, all_example_index)

    return dataset, features

def ignore_none_dontcare(pred_belief, target_belief):

    if 'not mentioned' in pred_belief or 'dontcare' in pred_belief or "none" in pred_belief:
        pred_belief = "none"

    if 'not mentioned' in target_belief or 'dontcare' in target_belief or "none" in target_belief:
        target_belief = "none"

    return pred_belief, target_belief


def cal_joint_acc(task_name,dataset_config, preds, args_ignore_none_and_dontcare):

    acc_list = []
    key_class_label_id = 'class_label_id_%s'
    key_class_prediction = 'class_prediction_%s'
    key_start_pos = 'start_pos_%s'
    key_start_prediction = 'start_prediction_%s'
    key_end_pos = 'end_pos_%s'
    key_end_prediction = 'end_prediction_%s'
    key_refer_id = 'refer_id_%s'
    key_refer_prediction = 'refer_prediction_%s'
    key_slot_groundtruth = 'slot_groundtruth_%s'
    key_slot_prediction = 'slot_prediction_%s'

    class_types, slots, label_maps = load_dataset_config(dataset_config)

    # Prepare label_maps
    label_maps_tmp = {}
    for v in label_maps:
        label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in label_maps[v]]
    label_maps = label_maps_tmp
    goal_correctness = 1.0
    pred_list = []
    gt_list = []
    slot_analysis = []
    slot_name_order = []
    for slot in slots:
        pred_flag,joint_pd_slot,joint_gt_slot = get_joint_slot_correctness(preds, class_types, label_maps,
                                             key_class_label_id=(key_class_label_id % slot),
                                             key_class_prediction=(key_class_prediction % slot),
                                             key_start_pos=(key_start_pos % slot),
                                             key_start_prediction=(key_start_prediction % slot),
                                             key_end_pos=(key_end_pos % slot),
                                             key_end_prediction=(key_end_prediction % slot),
                                             key_refer_id=(key_refer_id % slot),
                                             key_refer_prediction=(key_refer_prediction % slot),
                                             key_slot_groundtruth=(key_slot_groundtruth % slot),
                                             key_slot_prediction=(key_slot_prediction % slot)
                                             )



        if(args_ignore_none_and_dontcare):
            joint_pd_slot, joint_gt_slot = ignore_none_dontcare(pred_belief=joint_pd_slot,target_belief=joint_gt_slot)
            if(joint_pd_slot == joint_gt_slot):
                pred_flag = 1.0
        slot_analysis.append(pred_flag)
        slot_name_order.append(slot)
        goal_correctness *= pred_flag
        if((joint_pd_slot=="none") and (joint_gt_slot=="none")):
            continue
        else:
            pred_list.append(slot+"-"+str(joint_pd_slot))
            gt_list.append(slot+"-"+str(joint_gt_slot))

    return {"Joint Acc":goal_correctness,
            "Prediction":sorted(pred_list),
            "Ground Truth":sorted(gt_list),
            "Slot_analysis":slot_analysis,
            "Slot_order":slot_name_order}

class Parameters(object):

    def __init__(self):
        self.append_history=True
        self.checkpoint = MODEL_CHECKPOINT
        self.class_aux_feats_ds=True
        self.class_aux_feats_inform=True
        self.class_loss_ratio=0.8
        self.config_name=''
        self.data_dir='../trippy-public/data/MULTIWOZ2.1'
        self.dataset_config='../trippy-public/dataset_config/multiwoz21.json'
        self.delexicalize_sys_utts=True
        self.do_eval=True
        self.do_lower_case=True
        self.do_train=False
        self.dropout_rate=0.3 
        self.evaluate_during_training=False
        self.heads_dropout=0.0
        self.label_value_repetitions=True
        self.max_seq_length=180
        self.model_name_or_path='bert-base-uncased'
        self.model_type='bert'
        self.no_cuda=False
        self.output_dir= "/".join(MODEL_CHECKPOINT.split("/")[:-1])
        self.overwrite_cache=False
        self.overwrite_output_dir=False
        self.per_gpu_eval_batch_size=1 
        self.predict_type='test'
        self.refer_loss_for_nonpointable=False
        self.seed=42
        self.svd=0.0 
        self.swap_utterances=True
        self.task_name='multiwoz21gui'
        self.token_loss_for_nonpointable=False
        self.tokenizer_name=''
        self.use_history_labels=True
        self.n_gpu = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.device = device    

class TripPy_DST(object):
    """
    Update renormalize original input data problem
    """

    def __init__(self):

        args = Parameters()
        assert(args.svd >= 0.0 and args.svd <= 1.0)
        assert(args.class_aux_feats_ds is False or args.per_gpu_eval_batch_size == 1)
        assert(not args.class_aux_feats_inform or args.per_gpu_eval_batch_size == 1)
        assert(not args.class_aux_feats_ds or args.per_gpu_eval_batch_size == 1)
        self.task_name = "multiwoz21gui"
        if self.task_name not in PROCESSORS:
            raise ValueError("Task not found: %s" % (self.task_name))

        self.processor = PROCESSORS[self.task_name](args.dataset_config,args.data_dir)
        dst_slot_list = self.processor.slot_list
        dst_class_types = self.processor.class_types
        dst_class_labels = len(dst_class_types)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        set_seed(args)
        self.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        self.config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

        self.config.dst_dropout_rate = args.dropout_rate
        self.config.dst_heads_dropout_rate = args.heads_dropout
        self.config.dst_class_loss_ratio = args.class_loss_ratio
        self.config.dst_token_loss_for_nonpointable = args.token_loss_for_nonpointable
        self.config.dst_refer_loss_for_nonpointable = args.refer_loss_for_nonpointable
        self.config.dst_class_aux_feats_inform = args.class_aux_feats_inform
        self.config.dst_class_aux_feats_ds = args.class_aux_feats_ds
        self.config.dst_slot_list = dst_slot_list
        self.config.dst_class_types = dst_class_types
        self.config.dst_class_labels = dst_class_labels
        self.tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        self.model = model_class.from_pretrained(args.checkpoint)
        self.model.to(self.device)
        self.args = args
        self.domain_align = { 'attraction-area':"attraction-area",
                             'attraction-name':"attraction-name",
                             'attraction-type':"attraction-type",
                             'hotel-area':"hotel-area",
                             'hotel-book day':"hotel-book_day",
                             'hotel-book people':"hotel-book_people",
                             'hotel-book stay':"hotel-book_stay",
                             'hotel-internet':"hotel-internet",
                             'hotel-name':"hotel-name",
                             'hotel-parking':"hotel-parking",
                             'hotel-pricerange':"hotel-pricerange",
                             'hotel-stars':"hotel-stars", 
                             'hotel-type':"hotel-type",
                             'restaurant-area':"restaurant-area",
                             'restaurant-book day':"restaurant-book_day",
                             'restaurant-book people':"restaurant-book_people",
                             'restaurant-book time':"restaurant-book_time",
                             'restaurant-food': "restaurant-food",
                             'restaurant-name':"restaurant-name",
                             'restaurant-pricerange':"restaurant-pricerange",
                             'taxi-arriveby':"taxi-arriveBy",
                             'taxi-departure': 'taxi-departure',
                             'taxi-destination':'taxi-destination',
                             'taxi-leaveat':"taxi-leaveAt",
                             'train-arriveby':"train-arriveBy",
                             'train-book people':"train-book_people",
                             'train-day':'train-day',
                             'train-departure':"train-departure",
                             'train-destination':"train-destination",
                             'train-leaveat':"train-leaveAt"}

    def convert_turn_label(self,turn_label):

        new_turn_label = {}
        for slot,value in turn_label:
            slot = self.domain_align[slot]
            new_turn_label[slot] = value

        return new_turn_label

    def dst_query(self, dialog_id, turn_id, new_usr_utter,new_turn_label,args_ignore_none_and_dontcare):

        logging.disable(logging.CRITICAL)
        new_turn_label = self.convert_turn_label(new_turn_label)
        global_step = self.args.checkpoint.split('-')[-1]
        all_preds = evaluate(self.args, self.model, self.tokenizer, self.processor, dialog_id, turn_id, new_usr_utter,new_turn_label,prefix=global_step)
        result = cal_joint_acc(self.task_name,self.args.dataset_config,all_preds,args_ignore_none_and_dontcare)
        logging.disable(logging.NOTSET)
        return result




