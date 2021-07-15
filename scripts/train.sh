export CUDA_VISIBLE_DEVICES=2

cd /home/zhliu/MRC-NER/src

DATA_DIR="../data/dataset/zh_msra/"

python -u main.py \
    --train "${DATA_DIR}mrc-ner.train" \
    --dev "${DATA_DIR}mrc-ner.dev" \
    --test "${DATA_DIR}mrc-ner.test" \
    --save_model_dir "../save" \
    --bert_dir "/home/zhliu/plm/chinese_roberta_wwm_ext_large" \
    --log_path "../log" \
    --batch_size 24 \
    --span_loss_candidates "pred_and_gold" \
    --max_len 128 \
    --hidden_size 1024


find .. | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
