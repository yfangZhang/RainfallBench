export CUDA_VISIBLE_DEVICES=5

model_name=Koopa
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
  --e_layers 3 \
  --d_layers 3 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 1 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --target 'tp' \
  --itr 1 \
  --train_epochs 20 \
  --patience 20 \
  --inverse \
  --checkpoints long_term_forecast_rain_$seq_len'_'$pred_len'_'Koopa_custom_ftMS_sl24_ll12_pl4_dm512_nh8_el3_dl3_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0
done

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_24_6 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 24 \
#   --label_len 12 \
#   --pred_len 6 \
#   --e_layers 3 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 1 \
#   --des 'Exp' \
#   --learning_rate 0.001 \
#   --target 'tp' \
#   --itr 1 \
#   --train_epochs 20 \
#   --patience 20 \
#   --inverse \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_24_8 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 24 \
#   --label_len 12 \
#   --pred_len 8 \
#   --e_layers 3 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 1 \
#   --des 'Exp' \
#   --learning_rate 0.001 \
#   --target 'tp' \
#   --itr 1 \
#   --train_epochs 20 \
#   --patience 20 \
#   --inverse \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_24_10 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 24 \
#   --label_len 12 \
#   --pred_len 10 \
#   --e_layers 3 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 1 \
#   --des 'Exp' \
#   --learning_rate 0.001 \
#   --target 'tp' \
#   --itr 1 \
#   --train_epochs 20 \
#   --patience 20 \
#   --inverse \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_24_12 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 24 \
#   --label_len 12 \
#   --pred_len 12 \
#   --e_layers 3 \
#   --d_layers 3 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 1 \
#   --des 'Exp' \
#   --learning_rate 0.001 \
#   --target 'tp' \
#   --itr 1 \
#   --train_epochs 20 \
#   --patience 20 \
#   --inverse \