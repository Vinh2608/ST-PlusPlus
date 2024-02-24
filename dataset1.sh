#!/bin/bash
# Variables to define the configuration
dataset="dataset1"
split="1_4"
split_method="split_random"
model="unet"
backbone="resnet101"
mode="self_training" # Change to "self_training++" for the other mode
consistency="False" # Directly use this in the consistency_training argument

# Determine consistency training mode based on the consistency variable
if [ "$consistency" = "True" ]; then
    consistency_training="consistency_training"
else
    consistency_training="non_consistency_training"
fi

# Set the time phase based on your criteria, assuming it's manually set for now
time_phase="3" # This could be dynamically determined as needed

# Depending on the time_phase, set the time path
if [ "$time_phase" = "1" ]; then
    time_path="first_time"
elif [ "$time_phase" = "2" ]; then
    time_path="second_time"
else
    time_path="third_time" # Fallback case
fi

# Construct the semi_setting path including the time path
semi_setting="${dataset}/${split}/${split_method}/${model}/${backbone}/${mode}/${consistency_training}/${time_path}"

# Check if mode is self_training++
if [ "$mode" = "self_training++" ]; then
    plus_flag="--plus"
    reliable_id_path="--reliable-id-path outdir/reliable_ids/${semi_setting}"
else
    plus_flag=""
    reliable_id_path=""
fi

# Run the Python script with dynamic paths and arguments
python3 -W ignore main.py \
--dataset ${dataset} --config configs/${dataset}.yaml --data-root ./ \
--batch-size 16 --backbone ${backbone} --model ${model} \
--labeled-id-path dataset/splits/${dataset}/${split}/${split_method}/labeled.txt \
--unlabeled-id-path dataset/splits/${dataset}/${split}/${split_method}/unlabeled.txt \
--pseudo-mask-path outdir/pseudo_masks/${semi_setting} \
--save-path outdir/models/${semi_setting} \
--time ${time_phase} --consistency_training ${consistency} \
${plus_flag} ${reliable_id_path}
