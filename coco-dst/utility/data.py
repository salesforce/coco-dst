"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from utility.fix_label import fix_general_label_error
import numpy as np
import random

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
SLOT_MAPS = {
    "pricerange": "price range",
    "arriveby": "arrive by",
    "leaveat": "leave at"
}
ONTOLOGY = {
    "attraction": set(["area", "name", "type"]),
    "hotel": set(["area", "book day", "book people", "book stay", "internet", "name", "parking", "price range", "stars",
                  "type"]),
    "restaurant": set(["area", "book day", "book people", "book time", "food", "name", "price range"]),
    "taxi": set(["arrive by", "departure", "destination", "leave at"]),
    "train": set(["arrive by", "book people", "day", "departure", "destination", "leave at"])
}


class MultiWOZForT5(Dataset):

    def __init__(self, max_src_len, max_tgt_len, data_dir, tokenizer, shuffle_turn_label, data_ratio=1.0):

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.shuffle_turn_label = shuffle_turn_label
        self.data_ratio = data_ratio
        self.data = self.load_data(data_dir, data_ratio)

    def __getitem__(self, idx):

        data_detail = self.data[idx]
        sys = data_detail["system"].strip()
        user = data_detail["user"].strip()
        domain_slot_value_maps = data_detail["domain_slot_value_maps"]
        domain_slot_value_list = []
        for key, values in domain_slot_value_maps.items():
            for name, value in values:
                domain_slot_value_list.append(key + " " + name + " " + value)

        if (self.shuffle_turn_label):
            random.shuffle(domain_slot_value_list)

        domain_slot_value_str = " , ".join(domain_slot_value_list)

        src_text = "stu system: " + sys + " state: " + domain_slot_value_str + " </s>"
        tgt_text = user + " </s>"
        inputs = self.tokenizer.encode_plus(src_text, pad_to_max_length=True, truncation=True,
                                            max_length=self.max_src_len)
        targets = self.tokenizer.encode_plus(tgt_text, pad_to_max_length=True, truncation=True,
                                             max_length=self.max_tgt_len)

        return {"src_ids": inputs["input_ids"],
                "src_mask": inputs["attention_mask"],
                "tgt_ids": targets["input_ids"],
                "tgt_mask": targets["attention_mask"]}

    def __len__(self):
        return len(self.data)

    def load_data(self, file_name, data_ratio):
        with open(file_name) as f:
            data = []
            dials = json.load(f)
            if (data_ratio < 1.0):
                random.shuffle(dials)
                dials = dials[:int(len(dials) * data_ratio)]
            for dial_dict in dials:
                for turn in dial_dict["dialogue"]:
                    if (turn["domain"] not in EXPERIMENT_DOMAINS):
                        continue  # We skip turns that doesn't appear in EXPERIMENT_DOMAINS
                    domain_slot_value_maps = self.linear_turn_label(turn["turn_label"])
                    data_detail = {
                        "system": turn["system_transcript"],
                        "user": turn["transcript"],
                        "domain_slot_value_maps": domain_slot_value_maps,
                        "dialogue_idx": dial_dict["dialogue_idx"],
                        "turn_idx": turn["turn_idx"]
                    }
                    data.append(data_detail)
        return data

    def linear_turn_label(self, turn_label):

        domain_slot_value_maps = {}
        for (sub_domain, value) in turn_label:
            value = fix_general_label_error(sub_domain, value)
            if (value == "none"):
                continue
            cur_domain, slot_name = sub_domain.split("-")
            if (cur_domain not in EXPERIMENT_DOMAINS):
                return domain_slot_value_maps

            if (slot_name in SLOT_MAPS):
                slot_name = SLOT_MAPS[slot_name]

            if (cur_domain not in domain_slot_value_maps):
                domain_slot_value_maps[cur_domain] = [[slot_name, value]]
            else:
                domain_slot_value_maps[cur_domain].append([slot_name, value])

        return domain_slot_value_maps


class MultiWOZForT5_Interact(MultiWOZForT5):
    prompt_text = ""

    def __init__(self, data_dir, tokenizer, shuffle_turn_label):

        self.tokenizer = tokenizer
        self.shuffle_turn_label = shuffle_turn_label
        self.data = self.load_data(data_dir, 1.0)

    def __getitem__(self, idx):

        data_detail = self.data[idx]
        sys = data_detail["system"].strip()
        user = data_detail["user"].strip()

        if (not self.prompt_text):
            domain_slot_value_maps = data_detail["domain_slot_value_maps"]
        else:
            turn_label = []
            for ele in self.prompt_text.split(" , "):
                ele = ele.split("-")
                turn_label.append([ele[0] + "-" + ele[1], "-".join(ele[2:])])
            domain_slot_value_maps = self.linear_turn_label(turn_label)

        domain_slot_value_list = []
        for key, values in domain_slot_value_maps.items():
            for name, value in values:
                domain_slot_value_list.append(key + " " + name + " " + value)

        if (self.shuffle_turn_label):
            random.shuffle(domain_slot_value_list)

        domain_slot_value_str = " , ".join(domain_slot_value_list)
        src_text = "stu system: " + sys + " state: " + domain_slot_value_str + " </s>"
        input_ids = self.tokenizer(src_text)["input_ids"]

        return input_ids

    def get_dialID_turnID(self, idx):

        data_detail = self.data[idx]
        dialogue_idx = data_detail["dialogue_idx"].strip()
        turn_idx = int(data_detail["turn_idx"])
        return (dialogue_idx, turn_idx)

    def print_value(self, idx):

        data_detail = self.data[idx]
        sys = data_detail["system"].strip()
        user = data_detail["user"].strip()
        domain_slot_value_maps = data_detail["domain_slot_value_maps"]
        domain_slot_value_list = []
        for key, values in domain_slot_value_maps.items():
            for name, value in values:
                if (value != "none"):
                    domain_slot_value_list.append(key + "-" + name + "-" + value)

        domain_slot_value_str = " , ".join(domain_slot_value_list)

        print("Original System Utterance: ", sys)
        print("Original Turn-level belief state: ", domain_slot_value_str)
        print("Original User Utterance: ", user)


def get_dataloader(dataset, tokenizer, args, split='train'):
    def T5collate_fn(batch):
        """
        Modify target_id as label, T5 will modify label as valid taget input and add bos token
        """
        src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
        src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.long)
        tgt_ids = torch.tensor([example['tgt_ids'] for example in batch], dtype=torch.long)
        tgt_ids[tgt_ids[:, :] == 0] = -100
        tgt_mask = torch.tensor([example['tgt_mask'] for example in batch], dtype=torch.long)

        return {"src_ids": src_ids,
                "src_mask": src_mask,
                "tgt_ids": tgt_ids,
                "tgt_mask": tgt_mask}

    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(
            dataset if args.local_rank == -1 else DistributedSampler(dataset))  # SequentialSampler(dataset)
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=T5collate_fn)

    return dataloader, args
