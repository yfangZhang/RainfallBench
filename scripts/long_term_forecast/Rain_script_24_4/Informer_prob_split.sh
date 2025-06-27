export CUDA_VISIBLE_DEVICES=1

model_name=Informer_prob

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0  \
  --root_path ./dataset/Rain/ \
  --data_path JFNG_data_15min.csv \
  --model_id rain_24_4_MS_prob_cls_rec_loss2 \
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
  # --checkpoints '/root/zyf/Time-Series-Library-main/checkpoints/long_term_forecast_rain_24_4_MS_Informer_prob_Zero_y_Dataset_Custom_ftMS_sl24_ll12_pl4_dm512_nh8_el3_dl3_df2048_expand2_dc4_fc5_ebtimeF_dtTrue_Exp_0/checkpoint.pth' \