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

import json
import re
from tqdm import tqdm
from utils_dst import (DSTExample, convert_to_unicode)
import random

# Required for mapping slot names in dialogue_acts.json file
# to proper designations.
ACTS_DICT = {'taxi-depart': 'taxi-departure',
             'taxi-dest': 'taxi-destination',
             'taxi-leave': 'taxi-leaveAt',
             'taxi-arrive': 'taxi-arriveBy',
             'train-depart': 'train-departure',
             'train-dest': 'train-destination',
             'train-leave': 'train-leaveAt',
             'train-arrive': 'train-arriveBy',
             'train-people': 'train-book_people',
             'restaurant-price': 'restaurant-pricerange',
             'restaurant-people': 'restaurant-book_people',
             'restaurant-day': 'restaurant-book_day',
             'restaurant-time': 'restaurant-book_time',
             'hotel-price': 'hotel-pricerange',
             'hotel-people': 'hotel-book_people',
             'hotel-day': 'hotel-book_day',
             'hotel-stay': 'hotel-book_stay',
             'booking-people': 'booking-book_people',
             'booking-day': 'booking-book_day',
             'booking-stay': 'booking-book_stay',
             'booking-time': 'booking-book_time',
             }

LABEL_MAPS = {}  # Loaded from file


# Loads the dialogue_acts.json and returns a list
# of slot-value pairs.
def load_acts(input_file):
    with open(input_file) as f:
        acts = json.load(f)
    s_dict = {}
    for d in acts:
        for t in acts[d]:
            # Only process, if turn has annotation
            if isinstance(acts[d][t], dict):
                for a in acts[d][t]:
                    aa = a.lower().split('-')
                    if aa[1] == 'inform' or aa[1] == 'recommend' or aa[1] == 'select' or aa[1] == 'book':
                        for i in acts[d][t][a]:
                            s = i[0].lower()
                            v = i[1].lower().strip()
                            if s == 'none' or v == '?' or v == 'none':
                                continue
                            slot = aa[0] + '-' + s
                            if slot in ACTS_DICT:
                                slot = ACTS_DICT[slot]
                            key = d + '.json', t, slot
                            # In case of multiple mentioned values...
                            # ... Option 1: Keep first informed value
                            if key not in s_dict:
                                s_dict[key] = list([v])
                            # ... Option 2: Keep last informed value
                            # s_dict[key] = list([v])
    return s_dict


def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text)  # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5",
                  text)  # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text)  # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4",
                  text)  # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?",
                  lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) +
                            x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour
    return text


def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text)  # Systematic typo
    text = re.sub("guesthouse", "guest house", text)  # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text)  # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text)  # Normalization
    return text


# This should only contain label normalizations. All other mappings should
# be defined in LABEL_MAPS.
def normalize_label(slot, value_label):
    # Normalization of empty slots
    # import pdb;
    # pdb.set_trace()
    if value_label == '' or value_label == "not mentioned":
        return "none"

    # Normalization of time slots
    if "leaveAt" in slot or "arriveBy" in slot or slot == 'restaurant-book_time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)

    # Map to boolean slots
    if slot == 'hotel-parking' or slot == 'hotel-internet':
        if value_label == 'yes' or value_label == 'free':
            return "true"
        if value_label == "no":
            return "false"
    if slot == 'hotel-type':
        if value_label == "hotel":
            return "true"
        if value_label == "guest house":
            return "false"

    return value_label


def get_token_pos(tok_list, value_label):
    find_pos = []
    found = False
    label_list = [item for item in map(str.strip, re.split("(\W+)", value_label)) if len(item) > 0]
    len_label = len(label_list)
    for i in range(len(tok_list) + 1 - len_label):
        if tok_list[i:i + len_label] == label_list:
            find_pos.append((i, i + len_label))  # start, exclusive_end
            found = True
    return found, find_pos


def check_label_existence(value_label, usr_utt_tok):
    in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label)
    # If no hit even though there should be one, check for value label variants
    if not in_usr and value_label in LABEL_MAPS:
        for value_label_variant in LABEL_MAPS[value_label]:
            in_usr, usr_pos = get_token_pos(usr_utt_tok, value_label_variant)
            if in_usr:
                break
    return in_usr, usr_pos


def check_slot_referral(value_label, slot, seen_slots):
    referred_slot = 'none'
    if slot == 'hotel-stars' or slot == 'hotel-internet' or slot == 'hotel-parking':
        return referred_slot
    for s in seen_slots:
        # Avoid matches for slots that share values with different meaning.
        # hotel-internet and -parking are handled separately as Boolean slots.
        if s == 'hotel-stars' or s == 'hotel-internet' or s == 'hotel-parking':
            continue
        if re.match("(hotel|restaurant)-book_people", s) and slot == 'hotel-book_stay':
            continue
        if re.match("(hotel|restaurant)-book_people", slot) and s == 'hotel-book_stay':
            continue
        if slot != s and (slot not in seen_slots or seen_slots[slot] != value_label):
            if seen_slots[s] == value_label:
                referred_slot = s
                break
            elif value_label in LABEL_MAPS:
                for value_label_variant in LABEL_MAPS[value_label]:
                    if seen_slots[s] == value_label_variant:
                        referred_slot = s
                        break
    return referred_slot


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


def delex_utt(utt, values):
    utt_norm = tokenize(utt)
    for s, vals in values.items():
        for v in vals:
            if v != 'none':
                v_norm = tokenize(v)
                v_len = len(v_norm)
                for i in range(len(utt_norm) + 1 - v_len):
                    if utt_norm[i:i + v_len] == v_norm:
                        utt_norm[i:i + v_len] = ['[UNK]'] * v_len
    return utt_norm


# Fuzzy matching to label informed slot values
def check_slot_inform(value_label, inform_label):
    result = False
    informed_value = 'none'
    vl = ' '.join(tokenize(value_label))
    for il in inform_label:
        if vl == il:
            result = True
        elif is_in_list(il, vl):
            result = True
        elif is_in_list(vl, il):
            result = True
        elif il in LABEL_MAPS:
            for il_variant in LABEL_MAPS[il]:
                if vl == il_variant:
                    result = True
                    break
                elif is_in_list(il_variant, vl):
                    result = True
                    break
                elif is_in_list(vl, il_variant):
                    result = True
                    break
        elif vl in LABEL_MAPS:
            for value_label_variant in LABEL_MAPS[vl]:
                if value_label_variant == il:
                    result = True
                    break
                elif is_in_list(il, value_label_variant):
                    result = True
                    break
                elif is_in_list(value_label_variant, il):
                    result = True
                    break
        if result:
            informed_value = il
            break
    return result, informed_value


def get_turn_label(value_label, inform_label, sys_utt_tok, usr_utt_tok, slot, seen_slots, slot_last_occurrence):
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    informed_value = 'none'
    referred_slot = 'none'
    if value_label == 'none' or value_label == 'dontcare' or value_label == 'true' or value_label == 'false':
        class_type = value_label
    else:
        in_usr, usr_pos = check_label_existence(value_label, usr_utt_tok)
        if in_usr:
            class_type = 'copy_value'
            if slot_last_occurrence:
                (s, e) = usr_pos[-1]
                for i in range(s, e):
                    usr_utt_tok_label[i] = 1
            else:
                for (s, e) in usr_pos:
                    for i in range(s, e):
                        usr_utt_tok_label[i] = 1
        else:
            is_informed, informed_value = check_slot_inform(value_label, inform_label)
            if is_informed:
                class_type = 'inform'
            else:
                referred_slot = check_slot_referral(value_label, slot, seen_slots)
                if referred_slot != 'none':
                    class_type = 'refer'
                else:
                    class_type = 'unpointable'
    return informed_value, referred_slot, usr_utt_tok_label, class_type


def tokenize(utt):
    utt_lower = convert_to_unicode(utt).lower()
    utt_lower = normalize_text(utt_lower)
    utt_tok = [tok for tok in map(str.strip, re.split("(\W+)", utt_lower)) if len(tok) > 0]
    return utt_tok


def create_examples(input_file, acts_file, set_type, slot_list,
                    label_maps={},
                    append_history=False,
                    use_history_labels=False,
                    swap_utterances=False,
                    label_value_repetitions=False,
                    delexicalize_sys_utts=False,
                    analyze=False,
                    subset_dialog_file=""):
    """Read a DST json file into a list of DSTExample."""

    sys_inform_dict = load_acts(acts_file)

    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    if (subset_dialog_file):
        train_data = {}
        with open(subset_dialog_file, "r", encoding='utf-8') as reader:
            subset_dialog = json.load(reader)
        for turn_meta_data in subset_dialog:
            dial_ID = turn_meta_data["dialogue_idx"]
            if ((dial_ID not in train_data) and (
                    dial_ID in input_data)):  # (dial_ID not in train_data) should be removed?
                train_data[dial_ID] = input_data[dial_ID]
    else:
        train_data = input_data

    global LABEL_MAPS
    LABEL_MAPS = label_maps
    examples = []
    for dialog_id in tqdm(train_data):
        entry = train_data[dialog_id]
        utterances = entry['log']

        # Collects all slot changes throughout the dialog
        cumulative_labels = {slot: 'none' for slot in slot_list}

        # First system utterance is empty, since multiwoz starts with user input
        utt_tok_list = [[]]
        mod_slots_list = [{}]

        # Collect all utterances and their metadata
        usr_sys_switch = True
        turn_itr = 0
        for utt in utterances:

            # Assert that system and user utterances alternate
            is_sys_utt = utt['metadata'] != {}
            if usr_sys_switch == is_sys_utt:
                print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
                break
            usr_sys_switch = is_sys_utt

            if is_sys_utt:
                turn_itr += 1

            # Delexicalize sys utterance
            if delexicalize_sys_utts and is_sys_utt:
                inform_dict = {slot: 'none' for slot in slot_list}
                for slot in slot_list:
                    if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                        inform_dict[slot] = sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]
                utt_tok_list.append(delex_utt(utt['text'], inform_dict))  # normalize utterances
            else:
                utt_tok_list.append(tokenize(utt['text']))  # normalize utterances

            modified_slots = {}

            # If sys utt, extract metadata (identify and collect modified slots)
            if is_sys_utt:
                for d in utt['metadata']:
                    booked = utt['metadata'][d]['book']['booked']
                    booked_slots = {}
                    # Check the booked section
                    if booked != []:
                        for s in booked[0]:
                            booked_slots[s] = normalize_label('%s-%s' % (d, s), booked[0][s])  # normalize labels
                    # Check the semi and the inform slots
                    for category in ['book', 'semi']:
                        for s in utt['metadata'][d][category]:
                            cs = '%s-book_%s' % (d, s) if category == 'book' else '%s-%s' % (d, s)
                            value_label = normalize_label(cs, utt['metadata'][d][category][s])  # normalize labels
                            # Prefer the slot value as stored in the booked section
                            if s in booked_slots:
                                value_label = booked_slots[s]
                            # Remember modified slots and entire dialog state
                            if cs in slot_list and cumulative_labels[cs] != value_label:
                                modified_slots[cs] = value_label
                                cumulative_labels[cs] = value_label

            mod_slots_list.append(modified_slots.copy())
        # Form proper (usr, sys) turns

        turn_itr = 0
        diag_seen_slots_dict = {}
        diag_seen_slots_value_dict = {slot: 'none' for slot in slot_list}
        diag_state = {slot: 'none' for slot in slot_list}
        sys_utt_tok = []
        usr_utt_tok = []
        hst_utt_tok = []
        hst_utt_tok_label_dict = {slot: [] for slot in slot_list}
        for i in range(1, len(utt_tok_list) - 1, 2):
            sys_utt_tok_label_dict = {}
            usr_utt_tok_label_dict = {}
            value_dict = {}
            inform_dict = {}
            inform_slot_dict = {}
            referral_dict = {}
            class_type_dict = {}

            # Collect turn data
            if append_history:
                if swap_utterances:
                    hst_utt_tok = usr_utt_tok + sys_utt_tok + hst_utt_tok
                else:
                    hst_utt_tok = sys_utt_tok + usr_utt_tok + hst_utt_tok
            sys_utt_tok = utt_tok_list[i - 1]
            usr_utt_tok = utt_tok_list[i]
            turn_slots = mod_slots_list[i + 1]

            guid = '%s-%s-%s' % (set_type, str(dialog_id), str(turn_itr))

            if analyze:
                print("%15s %2s %s ||| %s" % (dialog_id, turn_itr, ' '.join(sys_utt_tok), ' '.join(usr_utt_tok)))
                print("%15s %2s [" % (dialog_id, turn_itr), end='')

            new_hst_utt_tok_label_dict = hst_utt_tok_label_dict.copy()
            new_diag_state = diag_state.copy()
            for slot in slot_list:
                value_label = 'none'
                if slot in turn_slots:
                    value_label = turn_slots[slot]
                    # We keep the original labels so as to not
                    # overlook unpointable values, as well as to not
                    # modify any of the original labels for test sets,
                    # since this would make comparison difficult.
                    value_dict[slot] = value_label
                elif label_value_repetitions and slot in diag_seen_slots_dict:
                    value_label = diag_seen_slots_value_dict[slot]

                # Get dialog act annotations
                inform_label = list(['none'])
                if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                    inform_label = list(
                        [normalize_label(slot, i) for i in sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]])
                elif (str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1]) in sys_inform_dict:
                    inform_label = list([normalize_label(slot, i) for i in sys_inform_dict[
                        (str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1])]])

                (informed_value,
                 referred_slot,
                 usr_utt_tok_label,
                 class_type) = get_turn_label(value_label,
                                              inform_label,
                                              sys_utt_tok,
                                              usr_utt_tok,
                                              slot,
                                              diag_seen_slots_value_dict,
                                              slot_last_occurrence=True)

                inform_dict[slot] = informed_value
                if informed_value != 'none':
                    inform_slot_dict[slot] = 1
                else:
                    inform_slot_dict[slot] = 0

                # Generally don't use span prediction on sys utterance (but inform prediction instead).
                sys_utt_tok_label = [0 for _ in sys_utt_tok]

                # Determine what to do with value repetitions.
                # If value is unique in seen slots, then tag it, otherwise not,
                # since correct slot assignment can not be guaranteed anymore.
                if label_value_repetitions and slot in diag_seen_slots_dict:
                    if class_type == 'copy_value' and list(diag_seen_slots_value_dict.values()).count(value_label) > 1:
                        class_type = 'none'
                        usr_utt_tok_label = [0 for _ in usr_utt_tok_label]

                sys_utt_tok_label_dict[slot] = sys_utt_tok_label
                usr_utt_tok_label_dict[slot] = usr_utt_tok_label

                if append_history:
                    if use_history_labels:
                        if swap_utterances:
                            new_hst_utt_tok_label_dict[slot] = usr_utt_tok_label + sys_utt_tok_label + \
                                                               new_hst_utt_tok_label_dict[slot]
                        else:
                            new_hst_utt_tok_label_dict[slot] = sys_utt_tok_label + usr_utt_tok_label + \
                                                               new_hst_utt_tok_label_dict[slot]
                    else:
                        new_hst_utt_tok_label_dict[slot] = [0 for _ in sys_utt_tok_label + usr_utt_tok_label +
                                                            new_hst_utt_tok_label_dict[slot]]

                # For now, we map all occurences of unpointable slot values
                # to none. However, since the labels will still suggest
                # a presence of unpointable slot values, the task of the
                # DST is still to find those values. It is just not
                # possible to do that via span prediction on the current input.
                if class_type == 'unpointable':
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                    if analyze:
                        if slot not in diag_seen_slots_dict or value_label != diag_seen_slots_value_dict[slot]:
                            print("(%s): %s, " % (slot, value_label), end='')
                elif slot in diag_seen_slots_dict and class_type == diag_seen_slots_dict[
                    slot] and class_type != 'copy_value' and class_type != 'inform':
                    # If slot has seen before and its class type did not change, label this slot a not present,
                    # assuming that the slot has not actually been mentioned in this turn.
                    # Exceptions are copy_value and inform. If a seen slot has been tagged as copy_value or inform,
                    # this must mean there is evidence in the original labels, therefore consider
                    # them as mentioned again.
                    class_type_dict[slot] = 'none'
                    referral_dict[slot] = 'none'
                else:
                    class_type_dict[slot] = class_type
                    referral_dict[slot] = referred_slot
                # Remember that this slot was mentioned during this dialog already.
                if class_type != 'none':
                    diag_seen_slots_dict[slot] = class_type
                    diag_seen_slots_value_dict[slot] = value_label
                    new_diag_state[slot] = class_type
                    # Unpointable is not a valid class, therefore replace with
                    # some valid class for now...
                    if class_type == 'unpointable':
                        new_diag_state[slot] = 'copy_value'

            if analyze:
                print("]")

            if swap_utterances:
                txt_a = usr_utt_tok
                txt_b = sys_utt_tok
                txt_a_lbl = usr_utt_tok_label_dict
                txt_b_lbl = sys_utt_tok_label_dict
            else:
                txt_a = sys_utt_tok
                txt_b = usr_utt_tok
                txt_a_lbl = sys_utt_tok_label_dict
                txt_b_lbl = usr_utt_tok_label_dict
            examples.append(DSTExample(
                guid=guid,
                text_a=txt_a,
                text_b=txt_b,
                history=hst_utt_tok,
                text_a_label=txt_a_lbl,
                text_b_label=txt_b_lbl,
                history_label=hst_utt_tok_label_dict,
                values=diag_seen_slots_value_dict.copy(),
                inform_label=inform_dict,
                inform_slot_label=inform_slot_dict,
                refer_label=referral_dict,
                diag_state=diag_state,
                class_label=class_type_dict))

            # Update some variables.
            hst_utt_tok_label_dict = new_hst_utt_tok_label_dict.copy()
            diag_state = new_diag_state.copy()

            turn_itr += 1

        if analyze:
            print("----------------------------------------------------------------------")

    return examples


def create_aug_examples(input_file, acts_file, set_type, slot_list,
                        label_maps={},
                        aug_file="",
                        append_history=False,
                        use_history_labels=False,
                        swap_utterances=False,
                        label_value_repetitions=False,
                        delexicalize_sys_utts=False,
                        analyze=False,
                        subset_dialog_file=""):
    examples = create_examples(input_file, acts_file, set_type, slot_list,
                               label_maps=label_maps,
                               append_history=append_history,
                               use_history_labels=use_history_labels,
                               swap_utterances=swap_utterances,
                               label_value_repetitions=label_value_repetitions,
                               delexicalize_sys_utts=delexicalize_sys_utts,
                               analyze=analyze,
                               subset_dialog_file=subset_dialog_file)

    DOMAIN_ALIGN = {'attraction-area': "attraction-area",
                    'attraction-name': "attraction-name",
                    'attraction-type': "attraction-type",
                    'hotel-area': "hotel-area",
                    'hotel-book day': "hotel-book_day",
                    'hotel-book people': "hotel-book_people",
                    'hotel-book stay': "hotel-book_stay",
                    'hotel-internet': "hotel-internet",
                    'hotel-name': "hotel-name",
                    'hotel-parking': "hotel-parking",
                    'hotel-pricerange': "hotel-pricerange",
                    'hotel-stars': "hotel-stars",
                    'hotel-type': "hotel-type",
                    'restaurant-area': "restaurant-area",
                    'restaurant-book day': "restaurant-book_day",
                    'restaurant-book people': "restaurant-book_people",
                    'restaurant-book time': "restaurant-book_time",
                    'restaurant-food': "restaurant-food",
                    'restaurant-name': "restaurant-name",
                    'restaurant-pricerange': "restaurant-pricerange",
                    'taxi-arriveby': "taxi-arriveBy",
                    'taxi-departure': 'taxi-departure',
                    'taxi-destination': 'taxi-destination',
                    'taxi-leaveat': "taxi-leaveAt",
                    'train-arriveby': "train-arriveBy",
                    'train-book people': "train-book_people",
                    'train-day': 'train-day',
                    'train-departure': "train-departure",
                    'train-destination': "train-destination",
                    'train-leaveat': "train-leaveAt"}

    def convert_turn_label(turn_label):
        new_turn_label = {}
        for slot, value in turn_label:
            if (slot in DOMAIN_ALIGN):
                slot = DOMAIN_ALIGN[slot]
                new_turn_label[slot] = value
        return new_turn_label

    sys_inform_dict = load_acts(acts_file)
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    file_train = '../multiwoz/MultiWOZ_2.1/train_dials.json'
    with open(file_train) as reader:
        train_data = json.load(reader)
    print(aug_file)
    with open(aug_file) as f:
        aug_dials = json.load(f)

    def get_CoCo_dialID_set(subset_dialog_file):

        with open(subset_dialog_file, "r", encoding='utf-8') as reader:
            subset_dialog = json.load(reader)

        CoCo_dialIDs = set()
        for turn_meta_data in subset_dialog:
            CoCo_dialIDs.add(turn_meta_data["dialogue_idx"])

        return CoCo_dialIDs

    # import pdb;
    # pdb.set_trace()
    if (subset_dialog_file):
        CoCo_dialIDs = get_CoCo_dialID_set(subset_dialog_file)
    else:
        CoCo_dialIDs = set()

    for dial_dict in tqdm(train_data):
        dialog_id = dial_dict["dialogue_idx"]
        if CoCo_dialIDs and (dialog_id not in CoCo_dialIDs):
            continue

        if (dialog_id not in input_data):
            continue

        for turn_id, turn in enumerate(dial_dict["dialogue"]):

            key = (dialog_id + str(turn_id))
            if (key in aug_dials):
                assert type(aug_dials[key]) == dict or type(aug_dials[key] == list)
                if (type(aug_dials[key]) == dict):
                    if (aug_dials[key]["success"]):
                        accu_examples = create_GUI_examples(input_data, dialog_id, turn_id, aug_dials[key]["new_utter"],
                                                            convert_turn_label(aug_dials[key]["new_turn_label"]),
                                                            sys_inform_dict, set_type, slot_list, label_maps=label_maps,
                                                            append_history=append_history,
                                                            use_history_labels=use_history_labels,
                                                            swap_utterances=swap_utterances,
                                                            label_value_repetitions=label_value_repetitions,
                                                            delexicalize_sys_utts=delexicalize_sys_utts,
                                                            analyze=analyze)
                    else:
                        accu_examples = create_GUI_examples(input_data, dialog_id, turn_id, turn["transcript"],
                                                            convert_turn_label(turn["turn_label"]), sys_inform_dict,
                                                            set_type, slot_list, label_maps=label_maps,
                                                            append_history=append_history,
                                                            use_history_labels=use_history_labels,
                                                            swap_utterances=swap_utterances,
                                                            label_value_repetitions=label_value_repetitions,
                                                            delexicalize_sys_utts=delexicalize_sys_utts,
                                                            analyze=analyze)
                    examples.append(accu_examples[-1])
                else:
                    for element in aug_dials[key]:
                        if (element["success"]):
                            accu_examples = create_GUI_examples(input_data, dialog_id, turn_id, element["new_utter"],
                                                                convert_turn_label(element["new_turn_label"]),
                                                                sys_inform_dict, set_type, slot_list,
                                                                label_maps=label_maps, append_history=append_history,
                                                                use_history_labels=use_history_labels,
                                                                swap_utterances=swap_utterances,
                                                                label_value_repetitions=label_value_repetitions,
                                                                delexicalize_sys_utts=delexicalize_sys_utts,
                                                                analyze=analyze)
                        else:
                            accu_examples = create_GUI_examples(input_data, dialog_id, turn_id, turn["transcript"],
                                                                convert_turn_label(turn["turn_label"]), sys_inform_dict,
                                                                set_type, slot_list, label_maps=label_maps,
                                                                append_history=append_history,
                                                                use_history_labels=use_history_labels,
                                                                swap_utterances=swap_utterances,
                                                                label_value_repetitions=label_value_repetitions,
                                                                delexicalize_sys_utts=delexicalize_sys_utts,
                                                                analyze=analyze)
                        examples.append(accu_examples[-1])

            # if(len(examples)>50):
            #     import pdb;
            #     pdb.set_trace()
    print(len(examples), " in total during training!")
    return examples


def create_GUI_examples(input_data, dialog_id, turn_id, new_usr_utter, new_turn_label, sys_inform_dict, set_type,
                        slot_list,
                        label_maps={},
                        append_history=False,
                        use_history_labels=False,
                        swap_utterances=False,
                        label_value_repetitions=False,
                        delexicalize_sys_utts=False,
                        analyze=False):
    global LABEL_MAPS
    LABEL_MAPS = label_maps

    examples = []

    entry = input_data[dialog_id]
    utterances = entry['log']

    # Collects all slot changes throughout the dialog
    cumulative_labels = {slot: 'none' for slot in slot_list}

    # First system utterance is empty, since multiwoz starts with user input
    utt_tok_list = [[]]
    mod_slots_list = [{}]

    # Collect all utterances and their metadata
    usr_sys_switch = True
    turn_itr = 0
    for utt in utterances:

        # Assert that system and user utterances alternate
        is_sys_utt = utt['metadata'] != {}
        if usr_sys_switch == is_sys_utt:
            print("WARN: Wrong order of system and user utterances. Skipping rest of dialog %s" % (dialog_id))
            break
        usr_sys_switch = is_sys_utt

        if is_sys_utt:
            turn_itr += 1

        # Delexicalize sys utterance
        if delexicalize_sys_utts and is_sys_utt:
            inform_dict = {slot: 'none' for slot in slot_list}
            for slot in slot_list:
                if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                    inform_dict[slot] = sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]
            utt_tok_list.append(delex_utt(utt['text'], inform_dict))  # normalize utterances
        elif ((2 * turn_id + 1) != len(utt_tok_list)):
            utt_tok_list.append(tokenize(utt['text']))  # normalize utterances
        else:
            # print(new_usr_utter)
            utt_tok_list.append(tokenize(new_usr_utter))  # normalize utterances

        modified_slots = {}

        # If sys utt, extract metadata (identify and collect modified slots)
        if is_sys_utt and ((2 * turn_id + 3) > len(utt_tok_list)):
            for d in utt['metadata']:
                booked = utt['metadata'][d]['book']['booked']
                booked_slots = {}
                # Check the booked section
                if booked != []:
                    for s in booked[0]:
                        booked_slots[s] = normalize_label('%s-%s' % (d, s), booked[0][s])  # normalize labels
                # Check the semi and the inform slots
                for category in ['book', 'semi']:
                    for s in utt['metadata'][d][category]:
                        cs = '%s-book_%s' % (d, s) if category == 'book' else '%s-%s' % (d, s)
                        value_label = normalize_label(cs, utt['metadata'][d][category][s])  # normalize labels
                        # Prefer the slot value as stored in the booked section
                        if s in booked_slots:
                            value_label = booked_slots[s]
                        # Remember modified slots and entire dialog state
                        if cs in slot_list and cumulative_labels[cs] != value_label:
                            modified_slots[cs] = value_label
                            cumulative_labels[cs] = value_label
        elif is_sys_utt:
            for slot, value in new_turn_label.items():
                new_value = normalize_label(slot, value)  # normalize labels
                modified_slots[slot] = new_value

        mod_slots_list.append(modified_slots.copy())

        if ((2 * turn_id + 3) == len(utt_tok_list)):
            break
    # Form proper (usr, sys) turns
    # import pdb;
    # pdb.set_trace()
    turn_itr = 0
    diag_seen_slots_dict = {}
    diag_seen_slots_value_dict = {slot: 'none' for slot in slot_list}
    diag_state = {slot: 'none' for slot in slot_list}
    sys_utt_tok = []
    usr_utt_tok = []
    hst_utt_tok = []
    hst_utt_tok_label_dict = {slot: [] for slot in slot_list}
    # print("--------------------------------")
    for i in range(1, len(utt_tok_list) - 1, 2):
        sys_utt_tok_label_dict = {}
        usr_utt_tok_label_dict = {}
        value_dict = {}
        inform_dict = {}
        inform_slot_dict = {}
        referral_dict = {}
        class_type_dict = {}

        # Collect turn data
        if append_history:
            if swap_utterances:
                hst_utt_tok = usr_utt_tok + sys_utt_tok + hst_utt_tok
            else:
                hst_utt_tok = sys_utt_tok + usr_utt_tok + hst_utt_tok
        sys_utt_tok = utt_tok_list[i - 1]
        usr_utt_tok = utt_tok_list[i]
        turn_slots = mod_slots_list[i + 1]
        # print("system: "," ".join(sys_utt_tok))
        # print("user: "," ".join(usr_utt_tok))
        # print("turn_label: ",turn_slots)
        guid = '%s-%s-%s' % (set_type, str(dialog_id), str(turn_itr))

        if analyze:
            print("%15s %2s %s ||| %s" % (dialog_id, turn_itr, ' '.join(sys_utt_tok), ' '.join(usr_utt_tok)))
            print("%15s %2s [" % (dialog_id, turn_itr), end='')

        new_hst_utt_tok_label_dict = hst_utt_tok_label_dict.copy()
        new_diag_state = diag_state.copy()
        for slot in slot_list:
            value_label = 'none'
            if slot in turn_slots:
                value_label = turn_slots[slot]
                # We keep the original labels so as to not
                # overlook unpointable values, as well as to not
                # modify any of the original labels for test sets,
                # since this would make comparison difficult.
                value_dict[slot] = value_label
            elif label_value_repetitions and slot in diag_seen_slots_dict:
                value_label = diag_seen_slots_value_dict[slot]

            # Get dialog act annotations
            inform_label = list(['none'])
            if (str(dialog_id), str(turn_itr), slot) in sys_inform_dict:
                inform_label = list(
                    [normalize_label(slot, i) for i in sys_inform_dict[(str(dialog_id), str(turn_itr), slot)]])
            elif (str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1]) in sys_inform_dict:
                inform_label = list([normalize_label(slot, i) for i in
                                     sys_inform_dict[(str(dialog_id), str(turn_itr), 'booking-' + slot.split('-')[1])]])

            (informed_value,
             referred_slot,
             usr_utt_tok_label,
             class_type) = get_turn_label(value_label,
                                          inform_label,
                                          sys_utt_tok,
                                          usr_utt_tok,
                                          slot,
                                          diag_seen_slots_value_dict,
                                          slot_last_occurrence=True)

            inform_dict[slot] = informed_value
            if informed_value != 'none':
                inform_slot_dict[slot] = 1
            else:
                inform_slot_dict[slot] = 0

            # Generally don't use span prediction on sys utterance (but inform prediction instead).
            sys_utt_tok_label = [0 for _ in sys_utt_tok]

            # Determine what to do with value repetitions.
            # If value is unique in seen slots, then tag it, otherwise not,
            # since correct slot assignment can not be guaranteed anymore.
            if label_value_repetitions and slot in diag_seen_slots_dict:
                if class_type == 'copy_value' and list(diag_seen_slots_value_dict.values()).count(value_label) > 1:
                    class_type = 'none'
                    usr_utt_tok_label = [0 for _ in usr_utt_tok_label]

            sys_utt_tok_label_dict[slot] = sys_utt_tok_label
            usr_utt_tok_label_dict[slot] = usr_utt_tok_label

            if append_history:
                if use_history_labels:
                    if swap_utterances:
                        new_hst_utt_tok_label_dict[slot] = usr_utt_tok_label + sys_utt_tok_label + \
                                                           new_hst_utt_tok_label_dict[slot]
                    else:
                        new_hst_utt_tok_label_dict[slot] = sys_utt_tok_label + usr_utt_tok_label + \
                                                           new_hst_utt_tok_label_dict[slot]
                else:
                    new_hst_utt_tok_label_dict[slot] = [0 for _ in sys_utt_tok_label + usr_utt_tok_label +
                                                        new_hst_utt_tok_label_dict[slot]]

            # For now, we map all occurences of unpointable slot values
            # to none. However, since the labels will still suggest
            # a presence of unpointable slot values, the task of the
            # DST is still to find those values. It is just not
            # possible to do that via span prediction on the current input.
            if class_type == 'unpointable':
                class_type_dict[slot] = 'none'
                referral_dict[slot] = 'none'
                if analyze:
                    if slot not in diag_seen_slots_dict or value_label != diag_seen_slots_value_dict[slot]:
                        print("(%s): %s, " % (slot, value_label), end='')
            elif slot in diag_seen_slots_dict and class_type == diag_seen_slots_dict[
                slot] and class_type != 'copy_value' and class_type != 'inform':
                # If slot has seen before and its class type did not change, label this slot a not present,
                # assuming that the slot has not actually been mentioned in this turn.
                # Exceptions are copy_value and inform. If a seen slot has been tagged as copy_value or inform,
                # this must mean there is evidence in the original labels, therefore consider
                # them as mentioned again.
                class_type_dict[slot] = 'none'
                referral_dict[slot] = 'none'
            else:
                class_type_dict[slot] = class_type
                referral_dict[slot] = referred_slot
            # Remember that this slot was mentioned during this dialog already.
            if class_type != 'none':
                diag_seen_slots_dict[slot] = class_type
                diag_seen_slots_value_dict[slot] = value_label
                new_diag_state[slot] = class_type
                # Unpointable is not a valid class, therefore replace with
                # some valid class for now...
                if class_type == 'unpointable':
                    new_diag_state[slot] = 'copy_value'

        if analyze:
            print("]")

        if swap_utterances:
            txt_a = usr_utt_tok
            txt_b = sys_utt_tok
            txt_a_lbl = usr_utt_tok_label_dict
            txt_b_lbl = sys_utt_tok_label_dict
        else:
            txt_a = sys_utt_tok
            txt_b = usr_utt_tok
            txt_a_lbl = sys_utt_tok_label_dict
            txt_b_lbl = usr_utt_tok_label_dict

        examples.append(DSTExample(
            guid=guid,
            text_a=txt_a,
            text_b=txt_b,
            history=hst_utt_tok,
            text_a_label=txt_a_lbl,
            text_b_label=txt_b_lbl,
            history_label=hst_utt_tok_label_dict,
            values=diag_seen_slots_value_dict.copy(),
            inform_label=inform_dict,
            inform_slot_label=inform_slot_dict,
            refer_label=referral_dict,
            diag_state=diag_state,
            class_label=class_type_dict))

        # Update some variables.
        hst_utt_tok_label_dict = new_hst_utt_tok_label_dict.copy()
        diag_state = new_diag_state.copy()
        if (turn_id == turn_itr):
            break
        turn_itr += 1

    if analyze:
        print("----------------------------------------------------------------------")

    return examples
