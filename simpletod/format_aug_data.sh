ori_dial_file="../multiwoz/MultiWOZ_2.1/train_dials.json"
aug_dials_file="../coco-dst/coco_data/coco-vs_rare_out_domain_train_classifier_change_add-2-max-3_drop-1_seed_0.json"
save_name="coco-vs_rare"
python format_aug_data.py --aug_dials_file=${aug_dials_file} --ori_dial_file=${ori_dial_file} --save_name=${save_name}