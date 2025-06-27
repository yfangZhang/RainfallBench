export CUDA_VISIBLE_DEVICES=7

model_name=SegRNN

seq_len=48
for pred_len in 4 6 8 10 12
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 24 \
  --pred_len $pred_len \
  --seg_len 2 \
  --enc_in 6 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --target 'tp' \
  --itr 1 \
  --train_epochs 20 \
  --patience 20 \
  --inverse 
done

