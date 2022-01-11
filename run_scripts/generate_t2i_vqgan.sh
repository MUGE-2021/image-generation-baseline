#!/usr/bin/env bash

DATA_DIR=../dataset/ECommerce-T2I
DICT_FILE=../utils/cn_tokenizer/dict.txt
SAVA_DIR=../checkpoints/ECommerce-T2I-vqgan
USER_DIR=../user_module
RESULT=../results/ECommerce-T2I-vqgan

CUDA_VISIBLE_DEVICES=0 python ../generate_t2i.py ${DATA_DIR} \
--user-dir ${USER_DIR} \
--dict-file ${DICT_FILE} \
--task t2i \
--path ${SAVA_DIR}/checkpoint_last.pt \
--beam 5 \
--batch-size 32 \
--max-len-b 256 \
--sampling \
--sampling-topk 100 \
--sampling-topp 0 \
--results-path ${RESULT} \
--caption-path "${DATA_DIR}/T2I_test.text.tsv" \
--image-path "${DATA_DIR}/T2I_test.tsv" \
--image-vocab-size 1024 \
--image-size 256 \
--seed 7 \
--num-workers 4