python deploy/python/predict.py --model_dir /home/public/rocketqa_model/static_graph/export_model_3050_v1.0 \
                                --query 签订日期 \
                                --max_seq_length 512 \
                                --batch_size 32 \
                                --device gpu \
                                --precision fp32 \
                                --cpu_threads 10 \
                                --save_log_path ./log_output/ \
                                --input_file ./workspace/data/test.tsv \
                                --model_name_or_path rocketqa-base-cross-encoder