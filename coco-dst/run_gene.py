"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
import numpy as np
import torch
from tqdm import tqdm
import sys
import re
import random
import copy
from utility.slot_value_ctrl import counterfactual_goal_generator, re_match
from utility.data import *
from utility.dictionary import *
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "t5": (T5ForConditionalGeneration, T5Tokenizer)
}


def set_seed(args_seed, args_n_gpu):
    np.random.seed(args_seed)
    torch.manual_seed(args_seed)
    random.seed(args_seed)
    if args_n_gpu > 0:
        torch.cuda.manual_seed_all(args_seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def build_dict(file_test):
    data_maps = {}
    with open(file_test) as f:
        data = json.load(f)
        for idx, dial in enumerate(data):
            data_maps[dial["dialogue_idx"]] = dial
            data_maps[str(idx)] = dial
            data_maps[idx] = dial

    return data_maps


def dial_print(dial):
    print("Dial domains: ", dial['domains'])
    print("-------------")
    for idx, turn in enumerate(dial['dialogue']):
        print("Turn ID: ", idx, " Turn Domain: ", turn["domain"])
        print("Bot: ", turn["system_transcript"])
        print("User: ", turn["transcript"])
        print("-------------")


def get_context(dial, turn_id):
    context = ""
    for idx, turn in enumerate(dial['dialogue']):
        if (idx <= turn_id):
            context += (" <system>: " + turn["system_transcript"] + " <user>: " + turn["transcript"])
        else:
            break

    return context


def turn_label2string(turn_label):
    label_list = []
    for (domain_slot, value) in turn_label:
        label_list.append(domain_slot + "-" + value)

    return " , ".join(label_list)


def para_filtering(turn, sentences, K):
    """filter paraphrase"""

    missing_values = []
    for domain_slot, value in turn["turn_label"]:
        if ((value in turn["system_transcript"]) and (value not in turn["transcript"])):
            missing_values.append(value)

    value_list = []
    best_sent = ""
    for domain_slot, value in turn["turn_label"]:

        domain, slot = domain_slot.split("-")
        if (slot == "parking"):
            value = slot
        elif (slot == "internet"):
            value = "wifi"
        if (value not in missing_values):
            value_list.append(value)

    count = 0
    for sent in sentences:
        sent = sent.lower()
        flag = True
        for value in value_list:
            if (value not in sent):
                flag = False
                break

        if (flag == True and (K == count)):
            best_sent = sent
            break
        elif (flag == True and (count < K)):
            count += 1

    return best_sent


def match_filtering(new_turn, ori_turn, sentences):
    ori_turn_label_set = set()
    for (slot, value) in ori_turn["turn_label"]:
        ori_turn_label_set.add(slot + "-" + value)

    new_turn_label_set = set()
    for (slot, value) in new_turn["turn_label"]:
        new_turn_label_set.add(slot + "-" + value)

    if (ori_turn_label_set == new_turn_label_set):
        return ""

    missing_values = []
    for domain_slot, value in ori_turn["turn_label"]:
        if ((value in ori_turn["system_transcript"]) and (value not in ori_turn["transcript"])):
            missing_values.append(value)

    value_list = []
    best_sent = ""
    for domain_slot, value in new_turn["turn_label"]:

        domain, slot = domain_slot.split("-")
        if (slot == "parking"):
            value = slot
        elif (slot == "internet"):
            value = "wifi"
        if (value not in missing_values):
            value_list.append(value)

    for sent in sentences:
        flag = True
        for value in value_list:
            if (value not in sent):
                flag = False
                break

        if (flag == True):
            best_sent = sent
            break

    return best_sent


def classifier_filtering(classifier_filter, dialogue_idx, turn_id, sentences, turn_label, thresh):
    """Use bert to get qualified candidates"""

    qual_sents = []
    flags = classifier_filter.query_filter(dialogue_idx, turn_id, sentences, turn_label, thresh)
    for idx, flag in enumerate(flags):
        if (flag):
            qual_sents.append(sentences[idx])

    return qual_sents


def subsitute(new_turn, ori_turn):
    """substitute value appear in the turn label
    """
    new_utter = (" " + ori_turn["transcript"] + " ")
    new_slot_dict = {}
    for domain_slot, value in new_turn["turn_label"]:
        new_slot_dict[domain_slot] = value

    for domain_slot, value in ori_turn["turn_label"]:

        search_span = re.search(r'[?,.! ]' + value + r'[?,.! ]', new_utter)
        if (search_span):
            new_value = new_slot_dict[domain_slot]
            new_utter = new_utter[:search_span.start() + 1] + new_value + new_utter[search_span.end() - 1:]

    new_utter = new_utter.strip()
    if (new_utter == ori_turn["transcript"]):
        return ""
    else:
        return new_utter


def filter_out_of_domain(turn):
    if (turn["domain"] not in EXPERIMENT_DOMAINS):
        return True
    else:
        for domain_slot, value in turn["turn_label"]:
            domain, _ = domain_slot.split("-")
            if (domain not in EXPERIMENT_DOMAINS):
                return True

    return False


def get_CoCo_dialID_set(subset_dialog_file):
    with open(subset_dialog_file, "r", encoding='utf-8') as reader:
        subset_dialog = json.load(reader)

    CoCo_dialIDs = set()
    for turn_meta_data in subset_dialog:
        CoCo_dialIDs.add(turn_meta_data["dialogue_idx"])

    return CoCo_dialIDs


def main():
    """-----------------------------------argument setting begins-----------------------------------------------"""
    args_seed = 0
    args_model_type = "t5"
    args_eval_data_file = "../multiwoz/MultiWOZ_2.1/train_dials.json"
    args_model_name_or_path = "./coco_model/model_1.0/checkpoint-12000"
    args_gene_data_save_dir = "./coco_data"
    args_subset_dialog_file = ""  # Subset of multiwoz used to train CoCo, "" means using whole data.
    args_length = 100
    args_stop_token = None
    args_temperature = 1.0
    args_repetition_penalty = 1.0
    args_k = 0
    args_num_beams = 5
    args_p = 1.0
    args_no_cuda = False
    args_thresh = 0.5
    args_do_sample = False
    args_num_return_sequences = args_num_beams
    args_shuffle_turn_label = True
    """-----------------------------------coco control settings ----------------------------------------------"""
    args_method = ["coco", "vs"]
    args_slot_value_dict = "out_domain_train"
    args_slot_combination_dict = "rare"
    args_classifier_filter = True
    args_change_slot = True
    args_add_slot = True
    args_drop_slot = True
    args_match_value = False
    args_max_drop = 1
    args_max_add = 2
    args_max_slot = 3
    num_aug_data = 8
    """-----------------------------------argument setting ends----------------------------------------------"""
    if (args_classifier_filter):
        print("Add classifier filter")
        from classifier_filter.bert_filter import BERTFilter
        classifier_filter = BERTFilter(args_eval_data_file)

    assert args_model_type == "t5"
    args_device = torch.device("cuda" if torch.cuda.is_available() and not args_no_cuda else "cpu")
    args_n_gpu = 0 if args_no_cuda else torch.cuda.device_count()
    set_seed(args_seed, args_n_gpu)
    try:
        args_model_type = args_model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args_model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args_model_name_or_path)
    model = model_class.from_pretrained(args_model_name_or_path)
    model.to(args_device)
    args_length = adjust_length_to_model(args_length, max_sequence_length=model.config.max_position_embeddings)

    dataset = MultiWOZForT5_Interact(data_dir=args_eval_data_file, tokenizer=tokenizer,
                                     shuffle_turn_label=args_shuffle_turn_label)
    data_maps = build_dict(args_eval_data_file)
    if (args_subset_dialog_file):
        CoCo_dialIDs = get_CoCo_dialID_set(args_subset_dialog_file)
    else:
        CoCo_dialIDs = set()
    success_gen = 0
    new_correct_pred = 0
    ori_correct_pred = 0
    save_info = {}
    global_idx = 0
    print(len(dataset), " data points in total")
    for num in tqdm(range(num_aug_data)):
        for idx in tqdm(range(len(dataset))):
            dialogue_idx, turn_idx = dataset.get_dialID_turnID(idx)
            if CoCo_dialIDs and (dialogue_idx not in CoCo_dialIDs):
                continue
            ori_turn = data_maps[dialogue_idx]["dialogue"][turn_idx]
            if (filter_out_of_domain(ori_turn)):
                continue
            new_turn = copy.deepcopy(ori_turn)
            success = False
            if (ori_turn["turn_label"]):
                if ("coco" in args_method):
                    new_turn = counterfactual_goal_generator(turn=new_turn, match_value=args_match_value,
                                                             change_slot=args_change_slot,
                                                             drop_slot=args_drop_slot, max_drop=args_max_drop,
                                                             add_slot=args_add_slot,
                                                             max_add=args_max_add, max_slot=args_max_slot,
                                                             slot_value_dict=SLOT_VALUE_DICT[args_slot_value_dict],
                                                             slot_occur_dict=SLOT_COMBINE_DICT[
                                                                 args_slot_combination_dict])
                    prompt_text = turn_label2string(new_turn["turn_label"])
                    dataset.prompt_text = prompt_text
                    encoded_prompt = dataset[idx]
                    encoded_prompt = torch.tensor(encoded_prompt, dtype=torch.long).view(1, -1)
                    encoded_prompt = encoded_prompt.to(args_device)
                    if encoded_prompt.size()[-1] == 0:
                        input_ids = None
                    else:
                        input_ids = encoded_prompt

                    output_sequences = model.generate(
                        input_ids=input_ids,
                        max_length=args_length + len(encoded_prompt[0]),
                        temperature=args_temperature,
                        top_k=args_k,
                        top_p=args_p,
                        num_beams=args_num_beams,
                        repetition_penalty=args_repetition_penalty,
                        do_sample=args_do_sample,
                        num_return_sequences=args_num_return_sequences,
                    )
                    # Remove the batch dimension when returning multiple sequences
                    if len(output_sequences.shape) > 2:
                        output_sequences.squeeze_()

                    generated_sequences = []
                    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                        generated_sequence = generated_sequence.tolist()
                        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                        text = text[:text.find(args_stop_token) if args_stop_token else None]
                        generated_sequences.append(text)

                    if (args_classifier_filter):
                        pre_filter_sequences = classifier_filtering(classifier_filter, dialogue_idx, turn_idx,
                                                                    generated_sequences, new_turn["turn_label"],
                                                                    args_thresh)
                        best_seq = match_filtering(new_turn, ori_turn, pre_filter_sequences)
                    else:
                        best_seq = match_filtering(new_turn, ori_turn, generated_sequences)

                    if (best_seq):
                        new_turn["transcript"] = best_seq
                        success = True
                    else:
                        new_turn = copy.deepcopy(ori_turn)

                if ("vs" in args_method and (not success)):
                    update_new_turn = counterfactual_goal_generator(turn=copy.deepcopy(new_turn), match_value=True,
                                                                    change_slot=True,
                                                                    drop_slot=False, max_drop=args_max_drop,
                                                                    add_slot=False,
                                                                    max_add=args_max_add, max_slot=args_max_slot,
                                                                    slot_value_dict=SLOT_VALUE_DICT[
                                                                        args_slot_value_dict],
                                                                    slot_occur_dict=SLOT_COMBINE_DICT[
                                                                        args_slot_combination_dict])

                    best_seq = subsitute(update_new_turn, new_turn)
                    if (best_seq):
                        success = True
                        update_new_turn["transcript"] = best_seq
                        new_turn = update_new_turn

            if (success):
                success_gen += 1
            else:
                new_turn = copy.deepcopy(ori_turn)

            global_idx += 1
            key = str(dialogue_idx) + str(turn_idx)
            ele = {"success": success,
                   "context": get_context(data_maps[dialogue_idx], turn_idx),
                   "new_utter": new_turn["transcript"],
                   "new_turn_label": new_turn["turn_label"],
                   "belief_state": new_turn["belief_state"]}

            if (key not in save_info):
                save_info[key] = [ele]
            else:
                save_info[key].append(ele)

            if (global_idx % 100 == 0):
                print("success generation rate: ", success_gen / global_idx)

    save_info["success rate"] = success_gen / (idx + 1)
    print("success generation rate: ", success_gen / (idx + 1))

    args_special_str = str(num_aug_data) + "_" + "-".join(
        args_method) + "_" + args_slot_combination_dict + "_" + args_slot_value_dict

    if (args_classifier_filter):
        args_special_str += "_classifier"

    if (args_match_value):
        args_special_str += "_match-value"

    if (args_change_slot):
        args_special_str += "_change"

    if (args_add_slot):
        args_special_str += ("_add-" + str(args_max_add) + "-max-" + str(args_max_slot))

    if (args_drop_slot):
        args_special_str += ("_drop-" + str(args_max_drop))

    saved_file_name = args_special_str + "_seed_" + str(args_seed) + ".json"

    if not os.path.exists(args_gene_data_save_dir):
        os.makedirs(args_gene_data_save_dir)

    with open(os.path.join(args_gene_data_save_dir, saved_file_name), "w") as f:
        json.dump(save_info, f, indent=4)


if __name__ == "__main__":
    main()
