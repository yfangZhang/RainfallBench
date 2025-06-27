export CUDA_VISIBLE_DEVICES=6

model_name=TimeXer
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_24_4 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 4 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --target 'tp' \
  --itr 1 \
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 512 \
#   --batch_size 4 \
#   --target 'tp' \
#   --itr 1 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 3 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 1024 \
#   --batch_size 4 \
#   --target 'tp' \
#   --itr 1 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --d_model 256 \
#   --batch_size 4 \
#   --target 'tp' \
#   --itr 1 \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --d_model 128 \
#   --batch_size 4 \
#   --target 'tp' \
#   --itr 1 \
