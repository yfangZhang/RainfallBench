export CUDA_VISIBLE_DEVICES=0

model_name=FilterTS
seq_len=24
for pred_len in 4 6 8 10 12
do
python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path ./dataset/Rain/ \
    --data_path JFNG_data_15min.csv \
    --model_id rain'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features MS \
    --freq h \
    --target 'tp' \
    --seq_len $seq_len \
    --label_len 12 \
    --pred_len $pred_len \
    --e_layers 2 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 6 \
    --des 'Exp' \
    --d_model 128 \
    --quantile 0.9 \
    --bandwidth 1 \
    --top_K_static_freqs 10 \
    --filter_type all \
    --learning_rate 0.005 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 20 \
    --patience 20 \
    --inverse \
    --checkpoints long_term_forecast_rain_$seq_len'_'$pred_len'_'FilterTS_custom_ftMS_sl24_ll12_pl4_dm128_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0

done