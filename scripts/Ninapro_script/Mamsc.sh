#!/bin/bash

learning_rates=(0.00006)
dropouts=(0.3)
seq_len=400
batch_size=32
model=Mamsc
data=ninapro
root_path=./dataset/Ninapro_DB2/
data_path="S1_E1_A1.mat,S2_E1_A1.mat,S3_E1_A1.mat"
enc_in=12
d_model=256
e_layers=2
d_layers=1
train_epochs=80
patience=8
gpu=0

if [[ "$data_path" == *"E1"* ]]; then
  c_out=17
elif [[ "$data_path" == *"E2"* ]]; then
  c_out=23
elif [[ "$data_path" == *"E3"* ]]; then
  c_out=9
else
  echo "Unknown data_path format: $data_path"
  exit 1
fi

for lr in "${learning_rates[@]}"; do
  for dropout in "${dropouts[@]}"; do
    exp_name="bs${batch_size}_lr${lr}_drop${dropout}"

    echo ">>>>>>>> Running: $exp_name >>>>>>"
    python run_classification.py \
      --is_training 1 \
      --model_id ${exp_name} \
      --model ${model} \
      --data ${data} \
      --root_path ${root_path} \
      --data_path ${data_path} \
      --seq_len ${seq_len} \
      --enc_in ${enc_in} \
      --c_out ${c_out} \
      --d_model ${d_model} \
      --e_layers ${e_layers} \
      --d_layers ${d_layers} \
      --dropout ${dropout} \
      --batch_size ${batch_size} \
      --learning_rate ${lr} \
      --train_epochs ${train_epochs} \
      --patience ${patience} \
      --gpu ${gpu} \
      --des Mamsc_sEMG \
      --itr 1
  done
done