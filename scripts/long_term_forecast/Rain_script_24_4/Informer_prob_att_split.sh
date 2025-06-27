export CUDA_VISIBLE_DEVICES=1

model_name=Informer_prob_zeroatt

seq_len=24
for pred_len in 4 6 8 10 12
do
python -u run_zero.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_$seq_len'_'$pred_len'_'MS_prob_onlyposatt \
  --model $model_name \
  --data Zero_y_Dataset_Custom \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 4 \
  --e_layers 3 \
  --d_layers 3 \
  --factor 5 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --target 'tp' \
  --itr 1 \
  --train_epochs 20 \
  --patience 20 \
  --inverse \

done