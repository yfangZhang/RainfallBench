# add --individual for P-sLSTM
export CUDA_VISIBLE_DEVICES=0
seq_len=24
for pred_len in  4 6 8 10 12
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_$seq_len'_'$pred_len \
  --model P_sLSTM \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 12 \
  --pred_len $pred_len \
  --target 'tp' \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --patch_size 2 --stride 2 \
  --num_blocks 2 \
  --channel 6 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 5 --train_epochs 20 --patience 20 \
  --inverse \
  --checkpoints /root/zyf/Time-Series-Library-main/checkpoints/long_term_forecast_rain_$seq_len'_'$pred_len'_'P_sLSTM_custom_ftMS_sl48_ll24_pl10_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth

done