export CUDA_VISIBLE_DEVICES=7
model_name=TimeMixerPP

seq_len=48
e_layers=3
down_sampling_layers=2
down_sampling_window=1
d_model=32
d_ff=64

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id Rain_$seq_len'_'4 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 12 \
  --pred_len 4 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window \
  --train_epochs 20 \
  --patience 20 \
  --inverse \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id Rain_$seq_len'_'6 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 12 \
  --pred_len 6 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window \
  --train_epochs 20 \
  --patience 20 \
  --inverse \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id Rain_$seq_len'_'8 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 12 \
  --pred_len 8 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window \
  --train_epochs 20 \
  --patience 20 \
  --inverse \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id Rain_$seq_len'_'10 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 12 \
  --pred_len 10 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window \
  --train_epochs 20 \
  --patience 20 \
  --inverse \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id Rain_$seq_len'_'12 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 12 \
  --pred_len 12 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window \
  --train_epochs 20 \
  --patience 20 \
  --inverse \