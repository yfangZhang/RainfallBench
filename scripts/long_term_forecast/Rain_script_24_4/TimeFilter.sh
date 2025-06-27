export CUDA_VISIBLE_DEVICES=3

model_name=TimeFilter

# 96
seq_len=24

for pred_len in 4 6 8 10 12
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path ./dataset/Rain/ \
    --data_path JFNG_data_15min.csv \
    --model_id rain_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features MS \
    --seq_len $seq_len \
    --label_len 12 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 6 \
    --patch_len 6 \
    --des 'Exp' \
    --target 'tp' \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.3 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 20 \
    --patience 20 \
    --inverse 
done