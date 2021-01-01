export CUDA_VISIBLE_DEVICES=0
MODEL="gpt2"
MODEL_NAME="gpt2"
BATCH=2
OUTPUT_DIR="coco-vs_rare"
TRAIN_FILE=./resources/train.coco-vs_rare_aug_history_belief
TEST_FILE=./resources/val.history_belief

mkdir -p $OUTPUT_DIR


python main.py \
--output_dir=$OUTPUT_DIR \
--model_type=$MODEL \
--model_name_or_path=$MODEL_NAME \
--do_train \
--train_data_file=$TRAIN_FILE \
--do_eval \
--eval_data_file=$TEST_FILE \
--evaluate_during_training \
--save_steps 5000 \
--logging_steps 5000 \
--per_gpu_train_batch_size $BATCH \
--num_train_epochs 30
