from utils.config import *
from models.TRADE import *
from utils.create_data import normalize
import logging
MODEL_PATH = "baseline/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.5342"
#MODEL_PATH= "coco-vs_rare/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.5335"
#Checkpoint you want to do evaluation


args["path"] = MODEL_PATH
directory = args['path'].split("/")
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
decoder = directory[1].split('-')[0] 
BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
args["decoder"] = decoder
args["HDD"] = HDD
args["path"] = "../trade-dst/"+MODEL_PATH
args["out_dir"] = "/".join(args["path"].split("/")[:3])

if args['dataset']=='multiwoz':
    from utils.utils_multiWOZ_DST import *
else:
    print("You need to provide the --dataset information")


class Trade_DST():
    def __init__(self):

        self.model = None

    def dst_query(self, dialogue, turn_ID, user_input, ignore_none_and_dontcare):

        logging.disable(logging.CRITICAL)
        test, lang, SLOTS_LIST, gating_dict, max_word = prepare_GUI_data_seq(dialogue, turn_ID, user_input, False, args['task'], False, batch_size=BSZ)
        if(self.model is None):
            self.model = globals()[decoder](
                int(HDD), 
                lang=lang, 
                path=args['path'], 
                task=args["task"], 
                lr=0, 
                dropout=0,
                slots=SLOTS_LIST,
                gating_dict=gating_dict,
                nb_train_vocab=max_word)
        result = self.model.query(test, 1e7, SLOTS_LIST[3],ignore_none_and_dontcare)
        result.update({"context":test.dataset[0]["context_plain"]})
        result["Ground Truth"] = sorted(result["Ground Truth"])
        result["Prediction"] = sorted(result["Prediction"])
        logging.disable(logging.NOTSET)
        return result
