"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import copy
import random
import re 

def re_match(utter,value):
    search_span = re.search(r'[?,.! ]'+value+r'[?,.! ]'," "+utter+" ")
    if(search_span):
        return True
    else:
        return False

def update_turn(turn,new_turn_label_dict):

    old_turn_label_dict = {}
    for (slot,value) in turn["turn_label"]:
        old_turn_label_dict[slot] = value

    new_belief = []
    slot_set = set()
    for bs in turn["belief_state"]:
        slot , value = bs["slots"][0]
        copy_bs = copy.deepcopy(bs)
        slot_set.add(slot)                                                 # Record Slot in Previous Turns
        if(slot in old_turn_label_dict and slot in new_turn_label_dict):   # Update Slot Value in Current Turns
            new_value = new_turn_label_dict[slot]
            copy_bs["slots"][0][1] = new_value
            if(new_value in turn["system_transcript"]):                     # TODO. This can be done by compare old_value with new_value
                copy_bs["act"] = "keep"
            else:
                copy_bs["act"] = "update"
            new_belief.append(copy_bs)

        elif(slot not in old_turn_label_dict):                              # Maintain Slot,value in previous turn
            copy_bs["act"] = "keep"
            new_belief.append(copy_bs)                  

    new_turn_label = []
    for slot,value in new_turn_label_dict.items():
        if(slot not in slot_set):                                           # New Added Slot,value in should not appear in current turn's bs.
            copy_bs = {"slots": [[slot,value]],"act": "add"}
            new_belief.append(copy_bs)

        new_turn_label.append([slot,value])


    turn["belief_state"] = new_belief
    turn["turn_label"] = new_turn_label
    return turn

def gen_time_pair():

    time_formats = ["am","pm","standard"]
    time_format = np.random.choice(time_formats,1)[0]
    if(time_format=="am" or time_format=="pm"):
        hour = random.randint(1,11)
        leave_min = random.randint(10,29)
        arrive_min = leave_min + random.randint(10,30)
        leave_time = str(hour)+":"+str(leave_min)+" "+time_format
        arrive_time = str(hour)+":"+str(arrive_min)+" "+time_format
    else:
        hour = random.randint(13,23)
        leave_min = random.randint(10,29)
        arrive_min = leave_min + random.randint(10,30)
        leave_time = str(hour)+":"+str(leave_min)
        arrive_time = str(hour)+":"+str(arrive_min)

    return(leave_time,arrive_time)

def fix_commonsense(turn_label_dict):

    if(("taxi-arriveby" in turn_label_dict) and ("taxi-leaveat" in turn_label_dict)):
        leave_time,arrive_time = gen_time_pair()
        turn_label_dict["taxi-leaveat"] = leave_time
        turn_label_dict["taxi-arriveby"] = arrive_time
    if(("train-arriveby" in turn_label_dict) and ("train-leaveat" in turn_label_dict)):
        leave_time,arrive_time = gen_time_pair()
        turn_label_dict["taxi-leaveat"] = leave_time
        turn_label_dict["taxi-arriveby"] = arrive_time

    return turn_label_dict



def counterfactual_goal_generator(turn, match_value, change_slot, drop_slot, max_drop , add_slot, max_add, max_slot,slot_value_dict,slot_occur_dict):


    turn_label_dict = {}
    drop_label_dict = {}
    added_num = 0
    user_utter = turn["transcript"]
    
    if drop_slot:

        turn_label = copy.deepcopy(turn["turn_label"])
        random.shuffle(turn_label)
        for (slot,value) in turn_label:
            if((value not in turn["system_transcript"]) and (len(turn_label) > max_drop)):    # if turn label is less or equal to max_drop, we cannot drop it.
                if((match_value and re_match(user_utter,value)) or (not match_value)):
                    if(len(drop_label_dict) < max_drop):
                        drop_label_dict[slot] = value
                    else:
                        turn_label_dict[slot] = value
                else:
                    turn_label_dict[slot] = value
            else:
                turn_label_dict[slot] = value

    else:
        for (slot,value) in turn["turn_label"]:
            turn_label_dict[slot] = value

    if change_slot:                                # if value appear in system utterance, we always keep it.

        for (slot,value) in turn_label_dict.items():
            if(value not in turn["system_transcript"]):
                if((match_value and re_match(user_utter,value)) or (not match_value)):
                    value_list = slot_value_dict[slot]
                    value = np.random.choice(value_list,1)[0]
            
            turn_label_dict[slot] = value

    if(add_slot):
        
        exist_state = set()
        turn_domain = turn["domain"]
        for state in turn["belief_state"]:
            domain_slot , value = state["slots"][0]
            if(domain_slot in drop_label_dict):    # For dropped values, we allow it to add it back
                continue
            exist_state.add(domain_slot)

        turn_label_slot = list(turn_label_dict.keys())
        for domain_slot, value_list in slot_value_dict.items():
            domain , slot = domain_slot.split("-")
            if((domain_slot not in exist_state) and (domain == turn_domain) and (added_num < max_add) and (len(turn_label_dict) < max_slot)):
                occur_flag = False 
                for key in turn_label_slot:
                    if(domain_slot in slot_occur_dict[key]):
                        occur_flag = True 
                    else:
                        occur_flag = False
                        break

                if(occur_flag):
                    value = np.random.choice(value_list,1)[0]
                    turn_label_dict[domain_slot] = value
                    added_num += 1

    if((added_num ==0) and add_slot and (drop_label_dict != {})):
        for slot, value in drop_label_dict.items():
            if(change_slot): # change value
                value_list = slot_value_dict[slot]
                value = np.random.choice(value_list,1)[0]
            
            turn_label_dict[slot] = value

    turn_label_dict = fix_commonsense(turn_label_dict)

    turn = update_turn(turn,turn_label_dict)

    return turn

