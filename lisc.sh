#!/bin/bash

# Base Configuration settings
dataset="lisc"
split="1_4"
split_method="split_random"
models=('deeplabv3plus')  # Allowed models
backbones=('resnet50' 'resnet101')  # Allowed backbones
modes=('self_training')  # Training modes
consistencies=('False')  # Consistency training settings
addition="noisy_investigate"
time_phases=('1' '2')  # Time phases

# Iterate over all configurations
for model in "${models[@]}"; do
    for backbone in "${backbones[@]}"; do
        # Check if model and backbone are compatible
        if [[ "$model" == "deeplabv2" && "$backbone" != "resnet101" ]]; then
            continue  # Skip incompatible backbone for deeplabv2
        fi
        for mode in "${modes[@]}"; do
            for consistency in "${consistencies[@]}"; do
                for time_phase in "${time_phases[@]}"; do
                    # Determine consistency training mode
                    consistency_training="non_consistency_training"
                    if [ "$consistency" = "True" ]; then
                        consistency_training="consistency_training"
                    fi

                    # Set the time path
                    time_path="undefined_time"
                    if [ "$time_phase" = "1" ]; then
                        time_path="first_time"
                    elif [ "$time_phase" = "2" ]; then
                        time_path="second_time"
                    fi

                    # Construct the semi_setting path
                    semi_setting="${dataset}/${split}/${split_method}/${model}/${backbone}/${mode}/${consistency_training}/${time_path}"
                    if [ "$addition" != "nothing" ]; then
                        semi_setting="${dataset}/${split}/${split_method}/${model}/${backbone}/${addition}/${mode}/${consistency_training}/${time_path}"
                    fi

                    # Configure additional flags for 'self_training++'
                    plus_flag=""
                    reliable_id_path=""
                    if [ "$mode" = "self_training++" ]; then
                        plus_flag="--plus"
                        reliable_id_path="--reliable-id-path outdir/reliable_ids/${semi_setting}"
                    fi

                    # Execute the Python command
                    echo "Running configuration: Model=$model, Backbone=$backbone, Mode=$mode, Consistency=$consistency, Time Phase=$time_phase"
                    python3 -W ignore main.py \
                    --dataset ${dataset} \
                    --addition ${addition} \
                    --config configs/${dataset}.yaml \
                    --data-root ./ \
                    --batch-size 16 \
                    --backbone ${backbone} \
                    --model ${model} \
                    --labeled-id-path dataset/splits/${dataset}/${split}/${split_method}/labeled.txt \
                    --unlabeled-id-path dataset/splits/${dataset}/${split}/${split_method}/unlabeled.txt \
                    --pseudo-mask-path outdir/pseudo_masks/${semi_setting} \
                    --save-path outdir/models/${semi_setting} \
                    --time ${time_phase} \
                    --consistency_training ${consistency} \
                    ${plus_flag} ${reliable_id_path}
                done
            done
        done
    done
done