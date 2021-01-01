export CUDA_VISIBLE_DEVICES=0
TASK="multiwoz21"
DATA_DIR="data/MULTIWOZ2.1"

# Project paths etc. ----------------------------------------------
root="../coco-dst/coco_data/"
aug_file=${root}"8times_coco-vs_rare_out_domain_train_classifier_change_add-2-max-3_drop-1_seed_0.json"
subset_dialog_file=""
OUT_DIR=coco-vs_rare_8times
if [ ! -d "${OUT_DIR}" ]; then
  mkdir -p ${OUT_DIR}
fi
# Main ------------------------------------------------------------
#
for step in train dev test; do
    args_add=""
    if [ "$step" = "train" ]; then
	args_add="--do_train --predict_type=dummy"
    elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
	args_add="--do_eval --predict_type=${step}"
    fi

    python3 run_dst.py \
	    --task_name=${TASK} \
	    --data_dir=${DATA_DIR} \
	    --dataset_config=dataset_config/${TASK}.json \
	    --model_type="bert" \
	    --model_name_or_path="bert-base-uncased" \
	    --do_lower_case \
	    --subset_dialog_file=${subset_dialog_file} \
	    --aug_file=${aug_file} \
	    --learning_rate=1e-4 \
	    --num_train_epochs=10 \
	    --max_seq_length=180 \
	    --per_gpu_train_batch_size=48 \
	    --per_gpu_eval_batch_size=1 \
	    --output_dir=${OUT_DIR} \
	    --save_epochs=2 \
	    --logging_steps=10 \
	    --warmup_proportion=0.1 \
	    --eval_all_checkpoints \
	    --adam_epsilon=1e-6 \
	    --label_value_repetitions \
	    --swap_utterances \
	    --append_history \
	    --use_history_labels \
	    --delexicalize_sys_utts \
	    --class_aux_feats_inform \
	    --class_aux_feats_ds \
	    ${args_add} \
	    2>&1 | tee ${OUT_DIR}/${step}.log

    if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
    	python3 metric_bert_dst.py \
    		${TASK} \
		dataset_config/${TASK}.json \
    		"${OUT_DIR}/pred_res.${step}*json" \
    		2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
    fi
done
