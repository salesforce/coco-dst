# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
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

import glob
import json
import sys
import numpy as np
import re


def load_dataset_config(dataset_config):
    with open(dataset_config, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    return raw_config['class_types'], raw_config['slots'], raw_config['label_maps']


def tokenize(text):
    if "\u0120" in text:
        text = re.sub(" ", "", text)
        text = re.sub("\u0120", " ", text)
        text = text.strip()
    return ' '.join([tok for tok in map(str.strip, re.split("(\W+)", text)) if len(tok) > 0])


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i:i + value_len] == value_list:
            found = True
            break
    return found


def check_slot_inform(value_label, inform_label, label_maps):
    # import pdb;
    # pdb.set_trace()
    value = inform_label
    if value_label == inform_label:
        value = value_label
    elif is_in_list(inform_label, value_label):
        value = value_label
    elif is_in_list(value_label, inform_label):
        value = value_label
    elif inform_label in label_maps:
        for inform_label_variant in label_maps[inform_label]:
            if value_label == inform_label_variant:
                value = value_label
                break
            elif is_in_list(inform_label_variant, value_label):
                value = value_label
                break
            elif is_in_list(value_label, inform_label_variant):
                value = value_label
                break
    elif value_label in label_maps:
        for value_label_variant in label_maps[value_label]:
            if value_label_variant == inform_label:
                value = value_label
                break
            elif is_in_list(inform_label, value_label_variant):
                value = value_label
                break
            elif is_in_list(value_label_variant, inform_label):
                value = value_label
                break
    return value


def get_joint_slot_correctness(preds, class_types, label_maps,
                               key_class_label_id='class_label_id',
                               key_class_prediction='class_prediction',
                               key_start_pos='start_pos',
                               key_start_prediction='start_prediction',
                               key_end_pos='end_pos',
                               key_end_prediction='end_prediction',
                               key_refer_id='refer_id',
                               key_refer_prediction='refer_prediction',
                               key_slot_groundtruth='slot_groundtruth',
                               key_slot_prediction='slot_prediction'):
    
    for pred in preds:
        guid = pred['guid']  # List: set_type, dialogue_idx, turn_idx
        turn_gt_class = pred[key_class_label_id]
        turn_pd_class = pred[key_class_prediction]
        gt_start_pos = pred[key_start_pos]
        pd_start_pos = pred[key_start_prediction]
        gt_end_pos = pred[key_end_pos]
        pd_end_pos = pred[key_end_prediction]
        gt_refer = pred[key_refer_id]
        pd_refer = pred[key_refer_prediction]
        gt_slot = pred[key_slot_groundtruth]
        pd_slot = pred[key_slot_prediction]
        gt_slot = tokenize(gt_slot)
        pd_slot = tokenize(pd_slot)

        # Make sure the true turn labels are contained in the prediction json file!
        joint_gt_slot = gt_slot
    
        if guid[-1] == '0': # First turn, reset the slots
            joint_pd_slot = 'none'

        # If turn_pd_class or a value to be copied is "none", do not update the dialog state.
        if turn_pd_class == class_types.index('none'):
            pass
        elif turn_pd_class == class_types.index('dontcare'):
            joint_pd_slot = 'dontcare'
        elif turn_pd_class == class_types.index('copy_value'):
            joint_pd_slot = pd_slot
        elif 'true' in class_types and turn_pd_class == class_types.index('true'):
            joint_pd_slot = 'true'
        elif 'false' in class_types and turn_pd_class == class_types.index('false'):
            joint_pd_slot = 'false'
        elif 'refer' in class_types and turn_pd_class == class_types.index('refer'):
            if pd_slot[0:3] == "§§ ":
                if pd_slot[3:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps)
            elif pd_slot[0:2] == "§§":
                if pd_slot[2:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps)
            elif pd_slot != 'none':
                joint_pd_slot = pd_slot
        elif 'inform' in class_types and turn_pd_class == class_types.index('inform'):
            if pd_slot[0:3] == "§§ ":
                if pd_slot[3:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps)
            elif pd_slot[0:2] == "§§":
                if pd_slot[2:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps)
            else:
                print("ERROR: Unexpected slot value format. Aborting.")
                exit()
        else:
            print("ERROR: Unexpected class_type. Aborting.")
            exit()


        if joint_gt_slot == joint_pd_slot:
            pred_flag = 1.0
        elif joint_gt_slot != 'none' and joint_gt_slot != 'dontcare' and joint_gt_slot != 'true' and joint_gt_slot != 'false' and joint_gt_slot in label_maps:
            no_match = True
            for variant in label_maps[joint_gt_slot]:
                if variant == joint_pd_slot:
                    no_match = False
                    pred_flag = 1.0
                    break
            if no_match:
                pred_flag = 0.0
        else:
            pred_flag = 0.0

    return (pred_flag,joint_pd_slot,joint_gt_slot)
