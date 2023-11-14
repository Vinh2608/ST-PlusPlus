#!/bin/bash
export semi_setting='dataset1/1_4/split_random/deeplabv3plus/self_training++/resnet101/consistency_training/second_time'
python3 -W ignore main.py \
--dataset dataset1 --config configs/dataset1.yaml --data-root ./  \
--batch-size 16 --backbone resnet101 --model deeplabv3plus \
--labeled-id-path dataset/splits/dataset1/1_4/split_random/labeled.txt \
--unlabeled-id-path dataset/splits/dataset1/1_4/split_random/unlabeled.txt \
--pseudo-mask-path outdir/pseudo_masks/$semi_setting \
--save-path outdir/models/$semi_setting --plus --reliable-id-path outdir/reliable_ids/$semi_setting
