import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import ipdb
from simpletod_utils.multiwoz.nlp import normalize_lexical, normalize_beliefstate
from simpletod_utils.dst import ignore_none_dontcare, default_cleaning, ignore_none, ignore_not_mentioned, \
    IGNORE_TURNS_TYPE2
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers.file_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_gpt2").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
MODEL_CHECKPOINT = "../simpletod/baseline/checkpoint-825000"
# MODEL_CHECKPOINT="../simpletod/coco-vs_rare/checkpoint-1000000"
##Checkpoint you want to do evaluation


class SimpleToD_DST(object):
    def __init__(self):
        self.model_checkpoint = MODEL_CHECKPOINT
        self.decoding = "greedy"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_checkpoint)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_checkpoint)
        self.model.eval()
        self.model.to('cuda')
        self.break_tokens = self.tokenizer.encode(self.tokenizer._eos_token) + self.tokenizer.encode(
            '?') + self.tokenizer.encode('!')
        self.MAX_LEN = self.model.config.n_ctx
        self.clean_tokens = ['<|endoftext|>']
        self.default_cleaning = True
        self.type2_cleaning = False

    def get_belief_new_dbsearch(self, sent):
        if '<|belief|>' in sent:
            tmp = sent.strip(' ').split('<|belief|>')[-1].split('<|endofbelief|>')[0]
        else:
            return []
        tmp = tmp.strip(' .,')
        tmp = tmp.replace('<|endofbelief|>', '')
        tmp = tmp.replace('<|endoftext|>', '')
        belief = tmp.split(',')
        new_belief = []
        for bs in belief:
            bs = bs.strip(' .,')
            if bs not in new_belief:
                new_belief.append(bs)
        return new_belief

    def convert_belief(self, belief):
        dic = {}
        for bs in belief:
            if bs in [' ', '']:
                continue
            domain = bs.split(' ')[0]
            slot = bs.split(' ')[1]
            if slot == 'book':
                slot = ' '.join(bs.split(' ')[1:3])
                value = ' '.join(bs.split(' ')[3:])
            else:
                value = ' '.join(bs.split(' ')[2:])
            if domain not in dic:
                dic[domain] = {}
            try:
                dic[domain][slot] = value
            except:
                print(domain)
                print(slot)
        return dic

    def dial2text(self, dialogue, turn_id, new_user_utter):

        for idx, turn in enumerate(dialogue["dialogue"]):
            if (turn_id == 0):
                context = ('<|user|> {}'.format(normalize_lexical(new_user_utter)))
                break
            elif (idx < turn_id):
                if (idx == 0):
                    context = '<|user|> {}'.format(normalize_lexical(turn["transcript"]))
                else:
                    context += (' <|system|> {}'.format(
                        normalize_lexical(turn["system_transcript"])) + ' <|user|> {}'.format(
                        normalize_lexical(turn["transcript"])))
            else:
                context += (' <|system|> {}'.format(
                    normalize_lexical(turn["system_transcript"])) + ' <|user|> {}'.format(
                    normalize_lexical(new_user_utter)))
                break
        text = '<|endoftext|> <|context|> {} <|endofcontext|>'.format(context)
        return text

    def cal_join_acc(self, turn_pred, turn_target, ignore_none_and_dontcare):

        joint_acc = 0.0
        for bs in turn_pred:
            if bs in self.clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                turn_pred.remove(bs)

        new_turn_pred = []
        for bs in turn_pred:
            for tok in self.clean_tokens:
                bs = bs.replace(tok, '').strip()
                new_turn_pred.append(bs)
        turn_pred = new_turn_pred

        if self.default_cleaning:
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

        if (ignore_none_and_dontcare):
            turn_pred, turn_target = ignore_none_dontcare(turn_pred, turn_target)
        else:
            turn_pred, turn_target = ignore_not_mentioned(turn_pred, turn_target)  # Adapted from original result

        join_flag = False
        if set(turn_target) == set(turn_pred):
            joint_acc = 1.0
            join_flag = True

        elif self.type2_cleaning:  # check for possible Type 2 noisy annotations
            flag = True
            for bs in turn_target:
                if bs not in turn_pred:
                    flag = False
                    break
            if flag:
                for bs in turn_pred:
                    if bs not in dialogue_target_final:
                        flag = False
                        break

            if flag:  # model prediction might be correct if found in Type 2 list of noisy annotations
                dial_name = dial.split('.')[0]
                if dial_name in IGNORE_TURNS_TYPE2 and turn_id in IGNORE_TURNS_TYPE2[dial_name]:  # ignore these turns
                    pass
                else:
                    joint_acc = 1.0
                    join_flag = True

        return {"Prediction": sorted(turn_pred),
                "Ground Truth": sorted(turn_target),
                "Joint Acc": joint_acc}

    def format_bs(self, target, pred):

        turn_pred = []
        turn_target = []
        for slot in target:
            slot, value = slot["slots"][0]
            value = normalize_beliefstate(value)
            slot = slot.replace("-", " ")
            turn_target.append(slot + " " + value)

        for domain in pred:
            for slot, value in pred[domain].items():
                turn_pred.append(domain + " " + slot + " " + value)

        return turn_target, turn_pred

    def dst_query(self, dialogue, turn_id, new_user_utter, ignore_none_and_dontcare):

        text = self.dial2text(dialogue, turn_id, new_user_utter)
        text = text.strip()
        indexed_tokens = self.tokenizer.encode(text)
        if len(indexed_tokens) > self.MAX_LEN:
            indexed_tokens = indexed_tokens[-1 * MAX_LEN:]
        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')
        predicted_index = indexed_tokens[-1]
        with torch.no_grad():
            while predicted_index not in self.break_tokens:
                outputs = self.model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]
                tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                if len(indexed_tokens) > self.MAX_LEN:
                    break
                if self.tokenizer.decode(indexed_tokens).endswith('<|endofbelief|>'):
                    break
        # ipdb.set_trace()
        tmp_pred = self.tokenizer.decode(indexed_tokens)
        try:
            pred_belief_text = self.get_belief_new_dbsearch(tmp_pred)
            pred_beliefs = self.convert_belief(pred_belief_text)
            turn_target, turn_pred = self.format_bs(target=dialogue["dialogue"][turn_id]["belief_state"],
                                                    pred=pred_beliefs)
            result = self.cal_join_acc(turn_pred=turn_pred, turn_target=turn_target,
                                       ignore_none_and_dontcare=ignore_none_and_dontcare)
            result["context"] = text
        except:
            return {"Prediction": [],
                    "Ground Truth": dialogue["dialogue"][turn_id]["belief_state"],
                    "Joint Acc": 0.0}

        return result
