#!/bin/bash
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1
python predict.py \
                --device 'gpu' \
                --params_path /home/public/rocketqa_model/dynamic_graph/model_10650_v1.1/model_state.pdparams \
                --model_name_or_path rocketqa-base-cross-encoder \
                --test_set /home/public/rocketqa_model/data/val.tsv \
                --topk 10 \
                --batch_size 128 \
                --max_seq_length 512