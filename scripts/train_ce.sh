#!/bin/bash
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "1,4,6,7" --log_dir="logs" train_ce.py \
        --device gpu \
        --train_set /home/public/rocketqa_model/data/train.tsv \
        --test_file /home/public/rocketqa_model/data/val.tsv \
        --save_dir workspace/checkpoints \
        --model_name_or_path rocketqa-base-cross-encoder \
        --batch_size 16 \
        --save_steps 10000 \
        --max_seq_len 512 \
        --learning_rate 1E-5 \
        --weight_decay  0.01 \
        --warmup_proportion 0.0 \
        --logging_steps 10 \
        --seed 1 \
        --epochs 50 \
        --eval_step 1000
