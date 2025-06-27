export CUDA_VISIBLE_DEVICES=1

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_24_6 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 1 \
  --batch_size 32 \
  --d_model 512 \
  --des 'Exp' \
  --target 'tp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'MSE' \
  --train_epochs 20 \
  --patience 5 \