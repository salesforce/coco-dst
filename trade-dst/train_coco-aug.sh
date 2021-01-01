export CUDA_VISIBLE_DEVICES=0
aug_file="../coco-dst/coco_data/coco-vs_rare_out_domain_train_classifier_change_add-2-max-3_drop-1_seed_0.json"
python3 myTrain.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 -patience=2 -out_dir="coco-vs_rare" -aug_file=${aug_file}
