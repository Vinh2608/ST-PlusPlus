#!/bin/bash
export semi_setting='raabin/1_4/split_random/deeplabv3plus/self_training/resnet50/non_consistency_training/first_time'
python3 -W ignore main.py \
--dataset raabin --config configs/raabin.yaml --data-root ./  \
--batch-size 16 --backbone resnet50 --model deeplabv3plus \
--labeled-id-path dataset/splits/raabin/1_4/split_random/labeled.txt \
--unlabeled-id-path dataset/splits/raabin/1_4/split_random/unlabeled.txt \
--pseudo-mask-path outdir/pseudo_masks/$semi_setting \
--save-path outdir/models/$semi_setting \
--time 1 --consistency_training False
 #--plus --reliable-id-path outdir/reliable_ids/$semi_setting \