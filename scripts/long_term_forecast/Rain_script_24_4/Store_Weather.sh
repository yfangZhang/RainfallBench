 #!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH

data_paths=("/root/zyf/Time-Series-Library-main/dataset/Rain/JFNG_data_15min.csv")
divides=("train" "val" "test")
num_nodes=6
input_len=24
output_len=4

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    log_file="./Results/emb_logs/${data_path}_${divide}.log"
    nohup python storage/store_emb.py --divide $divide --data_path $data_path --device $device --num_nodes $num_nodes --input_len $input_len --output_len $output_len > $log_file &
  done
done