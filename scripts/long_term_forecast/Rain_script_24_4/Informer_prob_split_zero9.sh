export CUDA_VISIBLE_DEVICES=1

model_name=Informer_prob

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path weather_data_with_zero_counts9.csv \
  --model_id rain_24_4_MS_prob_cls_rec_zero9 \
  --model $model_name \
  --data Zero_y_Dataset_Custom \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 4 \
  --e_layers 3 \
  --d_layers 3 \
  --factor 5 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --target 'tp' \
  --itr 1 \
  --train_epochs 20 \
  --patience 10\
  --inverse \