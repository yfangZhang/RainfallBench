export CUDA_VISIBLE_DEVICES=1

model_name=Informer_Patch

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path weather_data_with_zero_counts9.csv \
  --model_id rain_24_4_MS_zero_patch——permute0231 \
  --model $model_name \
  --data custom\
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 4 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --target 'tp' \
  --itr 1 \
  --train_epochs 20 \
  --patience 20 \