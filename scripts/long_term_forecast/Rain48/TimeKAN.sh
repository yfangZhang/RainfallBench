export CUDA_VISIBLE_DEVICES=7

model_name=TimeKAN

seq_len=48
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.005
d_model=16
d_ff=32
train_epochs=15
patience=15
begin_order=1
for pred_len in 4
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_$seq_len'_'$pred_len\
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
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
  --down_sampling_window $down_sampling_window\
  --begin_order $begin_order \
  --inverse 

done


