export CUDA_VISIBLE_DEVICES=0
python3 myTrain.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 -patience=2 -out_dir="baseline"
