# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from run_gene import *


def main():
    """-----------------------------------argument setting begins-----------------------------------------------"""
    args_seed = 0
    args_model_type = "t5"
    args_eval_data_file = "../multiwoz/MultiWOZ_2.1/test_dials.json"
    args_model_name_or_path = "./coco_model/checkpoint-12000"
    args_length = 100
    args_stop_token = None
    args_temperature = 1.0
    args_repetition_penalty = 1.0
    args_k = 0
    args_num_beams = 5
    args_p = 1.0
    args_seed = args_seed
    args_no_cuda = False
    args_thresh = 0.5
    args_do_sample = False
    args_num_return_sequences = args_num_beams
    args_shuffle_turn_label = True
    args_ignore_none_and_dontcare = True  # SimpleTOD uses this setting for evaluation.
    args_target_model = "trade"  # Possible models: ["trade","trippy","simpletod"]
    args_eval_result_save_dir = "./coco_eval/baseline"
    """-----------------------------------coco control settings ----------------------------------------------"""
    args_method = ["coco", "vs"]  # Possible models: [["coco"],["vs"],["coco","vs"]]
    args_slot_value_dict = "out_domain_test"  # "out_domain_train":OUT_DOMAIN_TRAIN_SLOT_VALUE_DICT,
    # "out_domain_test":OUT_DOMAIN_TEST_SLOT_VALUE_DICT,
    # "in_domain_test":IN_DOMAIN_TEST_SLOT_VALUE_DICT,
    args_slot_combination_dict = "rare"  # "freq":FREQ_SLOT_COMBINE_DICT, "neu":NEU_SLOT_COMBINE_DICT,  "rare":RARE_SLOT_COMBINE_DICT,
    args_classifier_filter = True
    args_change_slot = True
    args_add_slot = True
    args_drop_slot = True
    args_match_value = False
    args_max_drop = 1
    args_max_add = 2
    args_max_slot = 3
    """-----------------------------------argument setting ends----------------------------------------------"""
    if (args_classifier_filter):
        print("Add classifier filter")
        from classifier_filter.bert_filter import BERTFilter
        classifier_filter = BERTFilter(args_eval_data_file)

    if ("trade" in args_target_model):
        print("Evaluating Trade")
        sys.path.append("../trade-dst")
        from interface4eval_trade import Trade_DST
        target_model = Trade_DST()
    elif ("trippy" in args_target_model):
        print("Evaluating Trippy")
        sys.path.append("../trippy-public")
        from interface4eval_trippy import TripPy_DST
        target_model = TripPy_DST()
    elif ("simpletod" in args_target_model):
        print("Evaluating simpletod")
        sys.path.append("../simpletod")
        from interface4eval_simpletod import SimpleToD_DST
        target_model = SimpleToD_DST()

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
    success_gen = 0
    new_correct_pred = 0
    ori_correct_pred = 0
    save_info = {}
    print(len(dataset), " data points in total")
    for idx in tqdm(range(len(dataset))):

        dialogue_idx, turn_idx = dataset.get_dialID_turnID(idx)
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
                                                         slot_occur_dict=SLOT_COMBINE_DICT[args_slot_combination_dict])
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
                    text = text[: text.find(args_stop_token) if args_stop_token else None]
                    generated_sequences.append(text)
                    # print(text)
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
                                                                drop_slot=False, max_drop=args_max_drop, add_slot=False,
                                                                max_add=args_max_add, max_slot=args_max_slot,
                                                                slot_value_dict=SLOT_VALUE_DICT[args_slot_value_dict],
                                                                slot_occur_dict=SLOT_COMBINE_DICT[
                                                                    args_slot_combination_dict])
                best_seq = subsitute(update_new_turn, new_turn)
                if (best_seq):
                    success = True
                    update_new_turn["transcript"] = best_seq
                    new_turn = update_new_turn

            data_maps[dialogue_idx]["dialogue"][turn_idx] = new_turn
            if ("trade" in args_target_model):
                new_prediction = target_model.dst_query(data_maps[dialogue_idx], turn_idx, new_turn["transcript"],
                                                        args_ignore_none_and_dontcare)
            elif ("trippy" in args_target_model):
                new_prediction = target_model.dst_query(dialogue_idx, turn_idx, new_turn["transcript"],
                                                        new_turn["turn_label"], args_ignore_none_and_dontcare)
            elif ("simpletod" in args_target_model):
                new_prediction = target_model.dst_query(data_maps[dialogue_idx], turn_idx, new_turn["transcript"],
                                                        args_ignore_none_and_dontcare)

        data_maps[dialogue_idx]["dialogue"][turn_idx] = ori_turn
        if ("trade" in args_target_model):
            ori_prediction = target_model.dst_query(data_maps[dialogue_idx], turn_idx, ori_turn["transcript"],
                                                    args_ignore_none_and_dontcare)
        elif ("trippy" in args_target_model):
            ori_prediction = target_model.dst_query(dialogue_idx, turn_idx, ori_turn["transcript"],
                                                    ori_turn["turn_label"], args_ignore_none_and_dontcare)
        elif ("simpletod" in args_target_model):
            ori_prediction = target_model.dst_query(data_maps[dialogue_idx], turn_idx, ori_turn["transcript"],
                                                    args_ignore_none_and_dontcare)

        if (success):
            success_gen += 1
        else:
            new_prediction = ori_prediction
            new_turn = ori_turn

        new_correct_pred += new_prediction['Joint Acc']
        ori_correct_pred += ori_prediction['Joint Acc']
        save_info[str(dialogue_idx) + str(turn_idx)] = {"ori_prediction": ori_prediction,
                                                        "new_prediction": new_prediction,
                                                        "success": success,
                                                        "new_utter": new_turn["transcript"],
                                                        "new_turn_label": new_turn["turn_label"],
                                                        "ori_utter": ori_turn["transcript"],
                                                        "ori_turn_label": ori_turn["turn_label"]}

        if (idx % 100 == 0):
            print("success generation rate: ", success_gen / (idx + 1))
            print("avg new joint acc: ", new_correct_pred / (idx + 1))
            print("avg original joint acc: ", ori_correct_pred / (idx + 1))

    save_info["avg new joint acc"] = new_correct_pred / (idx + 1)
    save_info["avg ori joint acc"] = ori_correct_pred / (idx + 1)
    save_info["success rate"] = success_gen / (idx + 1)

    args_special_str = ("-".join(args_method) + "_" + args_slot_combination_dict + "_" + args_slot_value_dict)

    if (args_classifier_filter):
        args_special_str += "_classifier"

    if (args_ignore_none_and_dontcare):
        args_special_str += "_ignore"

    if (args_match_value):
        args_special_str += "_match-value"

    if (args_change_slot):
        args_special_str += "_change"

    if (args_add_slot):
        args_special_str += ("_add-" + str(args_max_add) + "-max-" + str(args_max_slot))

    if (args_drop_slot):
        args_special_str += ("_drop-" + str(args_max_drop))

    save_path = os.path.join(args_eval_result_save_dir, args_target_model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    saved_file_name = args_special_str + "_seed_" + str(args_seed) + ".json"
    with open(os.path.join(save_path, saved_file_name), "w") as f:
        json.dump(save_info, f, indent=4)


if __name__ == "__main__":
    main()
