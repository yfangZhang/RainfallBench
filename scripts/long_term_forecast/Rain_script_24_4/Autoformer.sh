export CUDA_VISIBLE_DEVICES=1

model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_24_4 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 4 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --itr 1 \