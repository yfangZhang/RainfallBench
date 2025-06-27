export CUDA_VISIBLE_DEVICES=6

seq_len=24
label_len=12
model_name=MSGNet

for pred_len in 4 6
do
python -u run.py \
    --is_training 0  \
    --task_name long_term_forecast \
    --root_path ./dataset/Rain/ \
    --data_path JFNG_data_15min.csv \
    --model_id rain'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features MS \
    --freq h \
    --target 'tp' \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 6 \
    --des 'Exp' \
    --d_model 64 \
    --d_ff 128 \
    --top_k 5 \
    --conv_channel 32 \
    --skip_channel 32 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 20 \
    --patience 20 \
    --inverse \

done