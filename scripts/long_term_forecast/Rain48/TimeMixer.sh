export CUDA_VISIBLE_DEVICES=7

model_name=TimeMixer

seq_len=48
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=64
d_ff=64
batch_size=32
train_epochs=20
patience=20

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_48_4 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len $seq_len \
#   --label_len 24 \
#   --pred_len 4 \
#   --e_layers $e_layers \
#   --d_layers 3 \
#   --factor 5 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --target 'tp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 32 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --inverse \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_48_6 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len $seq_len \
#   --label_len 24 \
#   --pred_len 6 \
#   --e_layers $e_layers \
#   --d_layers 3 \
#   --factor 5 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --target 'tp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 32 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --inverse \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_48_8 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 24 \
  --pred_len 8 \
  --e_layers $e_layers \
  --d_layers 3 \
  --factor 5 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 32 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --inverse \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_48_10 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len $seq_len \
#   --label_len 24 \
#   --pred_len 10 \
#   --e_layers $e_layers \
#   --d_layers 3 \
#   --factor 5 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --target 'tp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 32 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
  # --inverse \

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1  \
#   --root_path ./dataset/Rain/ \
#   --data_path JFNG_data_15min.csv \
#   --model_id rain_48_12 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len $seq_len \
#   --label_len 24 \
#   --pred_len 12 \
#   --e_layers $e_layers \
#   --d_layers 3 \
#   --factor 5 \
#   --enc_in 6 \
#   --dec_in 6 \
#   --c_out 6 \
#   --des 'Exp' \
#   --target 'tp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size 32 \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window \
#   --inverse \