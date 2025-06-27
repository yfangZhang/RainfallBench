export CUDA_VISIBLE_DEVICES=2
model_name=xPatch
seq_len=24
ma_type=reg
alpha=0.3
beta=0.3

for pred_len in  4 6 8 10 12
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0  \
    --root_path ./dataset/Rain/ \
    --data_path JFNG_data_15min.csv \
    --model_id rain_$seq_len'_'$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features MS \
    --seq_len $seq_len \
    --label_len 12 \
    --pred_len $pred_len \
    --enc_in 6 \
    --des 'Exp' \
    --target 'tp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta \
    --train_epochs 20 \
    --patience 20 \
    --inverse \

done