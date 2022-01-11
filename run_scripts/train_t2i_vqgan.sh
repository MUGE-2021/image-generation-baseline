TOTAL_NUM_UPDATES=45000
WARMUP_UPDATES=2700
LR=1e-04
BATCH_SIZE=4
UPDATE_FREQ=2
ARCH=t2i_baseline_base
DICT_FILE=../utils/cn_tokenizer/dict.txt
VOCAB_FILE=../utils/cn_tokenizer/vocab.txt
DATA_DIR=../dataset/ECommerce-T2I
SAVA_DIR=../checkpoints/ECommerce-T2I-vqgan
USER_DIR=../user_module

python ../train.py ${DATA_DIR} \
    --dict-file ${DICT_FILE} \
    --vocab-file ${VOCAB_FILE} \
    --save-dir ${SAVA_DIR} \
    --batch-size ${BATCH_SIZE} \
    --task t2i \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --layernorm-embedding \
    --share-decoder-input-output-embed \
    --arch ${ARCH} \
    --criterion cross_entropy_extend \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr ${LR} --total-num-update ${TOTAL_NUM_UPDATES} --warmup-updates ${WARMUP_UPDATES} \
    --update-freq ${UPDATE_FREQ} \
    --find-unused-parameters \
    --user-dir ${USER_DIR} \
    --log-format 'simple' --log-interval 10 \
    --fixed-validation-seed 7 \
    --save-interval 1 --validate-interval 1 \
    --max-update ${TOTAL_NUM_UPDATES} \
    --num-workers 4 \
    --image-vocab-size 1024 \
    --image-size 256 \
    --code-image-size 256 \
    --vae-model-type vqgan \

