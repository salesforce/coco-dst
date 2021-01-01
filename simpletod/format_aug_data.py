from tqdm import tqdm
import torch
import json
import ipdb
from simpletod_utils.multiwoz.nlp import normalize_lexical, normalize_beliefstate
from simpletod_utils.dst import ignore_none_dontcare, default_cleaning, IGNORE_TURNS_TYPE2
import logging
import argparse
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers.file_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_gpt2").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


ONTOLOGY = {
    "attraction":set(["area", "name", "type"]),
    "hotel": set(["area","book day","book people", "book stay", "internet", "name","parking", "pricerange", "stars","type"]),
    "restaurant": set(["area","book day","book people","book time","food","name","pricerange"]),
    "taxi": set(["arriveby","departure", "destination", "leaveat"]),
    "train":set(["arriveby","book people","day","departure","destination","leaveat"])
}
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
BELIEF_ORDER = {'taxi': ['leaveat','destination','departure','arriveby'],
                'police': [],
                'restaurant': ['food', 'pricerange', 'name', 'area','book time', 'book day', 'book people'],
                'hospital': ['department'],
                'hotel':['name','area','parking','pricerange','stars','internet','type','book stay', 'book day', 'book people'],
                'attraction': ['type', 'name', 'area'],
                'train': ['leaveat','destination','day','arriveby','departure','book people']}

def dial2text(dialogue, turn_id, new_user_utter,belief_state):
    
    for idx,turn in enumerate(dialogue["dialogue"]):
        if(turn_id==0):
            context =('<|user|> {}'.format(normalize_lexical(new_user_utter)))
            break
        elif(idx < turn_id):
            if(idx==0):
                context = '<|user|> {}'.format(normalize_lexical(turn["transcript"]))
            else:
                context += (' <|system|> {}'.format(normalize_lexical(turn["system_transcript"]))+' <|user|> {}'.format(normalize_lexical(turn["transcript"])))
        else:
            context +=(' <|system|> {}'.format(normalize_lexical(turn["system_transcript"]))+ ' <|user|> {}'.format(normalize_lexical(new_user_utter)))
            break
    context_text = '<|endoftext|> <|context|> {} <|endofcontext|> '.format(context)

    if len(belief_state) == 0:
        belief_state.append(' ')
    
    bs_text = '<|belief|> {} <|endofbelief|> <|endoftext|>'.format(' , '.join(belief_state))

    return context_text + bs_text

def format_bs(belief_state):

    domain_slot_value_maps = {}
    for slot_info in belief_state:
        domain_slot , value = slot_info["slots"][0]
        value = normalize_beliefstate(value)
        domain,slot = domain_slot.split("-")
        if(domain not in domain_slot_value_maps):
            domain_slot_value_maps[domain] = [[slot,value]]
        else:
            domain_slot_value_maps[domain].append([slot,value])

    for domain, slot_value_list in domain_slot_value_maps.items():

        existed_slot = set()
        book_flag = False
        for slot_name,value in slot_value_list:
            if("book" in slot_name):
                book_flag = True
            existed_slot.add(slot_name)

        if(domain not in ONTOLOGY):
            continue

        for new_slot in ONTOLOGY[domain].difference(existed_slot):
            if("book" in new_slot and book_flag):
                domain_slot_value_maps[domain].append([new_slot,"not mentioned"])
            elif("book" not in new_slot):
                domain_slot_value_maps[domain].append([new_slot,"not mentioned"])

    format_bs = []
    def find_value(slot_name,slot_value_list):
        for slot_value in slot_value_list:
            if(slot_name == slot_value[0]):
                return slot_value[1]
        return None

    for domain, slot_list in BELIEF_ORDER.items():
        if(domain in domain_slot_value_maps):
            slot_value_list = domain_slot_value_maps[domain]
            for slot in slot_list:
                value = find_value(slot,slot_value_list)
                if(value):
                    format_bs.append(domain+" "+slot+" "+value)

    return format_bs


def convert_data(dialogue, turn_id,new_user_utter,belief_state):
    
    belief_state = format_bs(belief_state)
    text = dial2text(dialogue,turn_id,new_user_utter,belief_state)
    text = text.strip()
    return  text

def gene_data(aug_dials_file,ori_dial_file,aug_type):

    data = []
    with open(aug_dials_file) as f:
        aug_dials = json.load(f)
    with open(ori_dial_file) as f:
        dials = json.load(f)
        for dial_dict in tqdm(dials):
            for ti, turn in enumerate(dial_dict["dialogue"]):
                data_point = convert_data(dial_dict,ti,turn["transcript"],turn["belief_state"])
                data.append(data_point+"\n")
                key = (dial_dict["dialogue_idx"]+str(ti))
                if(key in aug_dials and aug_dials[key]["success"]):
                    data_point = convert_data(dial_dict,ti,aug_dials[key]["new_utter"],aug_dials[key]["belief_state"])
                data.append(data_point+"\n")

    new_data_file = open("resources/train."+aug_type+"_aug_history_belief","w")
    for line in data:
        new_data_file.write(line)
    new_data_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_dials_file', type=str, help="file for augmentation")
    parser.add_argument('--ori_dial_file', type=str, default="",
                        help="original file")
    parser.add_argument('--save_name', type=str, default="",
                        help="save file name")
    args = parser.parse_args()
    print(args)
    gene_data(args.aug_dials_file,args.ori_dial_file,args.save_name)



