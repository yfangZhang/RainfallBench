export CUDA_VISIBLE_DEVICES=0

seq_len=24
model=GPT4TS

for percent in 100
do
for pred_len in 4 6 8 10 12
do

python GPT4TS_main.py \
    --task_name long_term_forecast \
    --root_path ./dataset/Rain/ \
    --data_path JFNG_data_15min.csv \
    --model_id rain_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 12 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --lradj type3 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --is_gpt 1 \
    --train_epochs 20 \
    --patience 20 \
    
done
done