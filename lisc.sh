export semi_setting='lisc/1_4/split_random/deeplabv3plus/self_training/resnet101/increased_contrast/non_consistency_training/second_time'
python3 -W ignore main.py \
--dataset lisc --config configs/lisc.yaml --data-root ./  \
--batch-size 16 --backbone resnet101 --model deeplabv3plus \
--labeled-id-path dataset/splits/lisc/1_4/split_kmeans/labeled.txt \
--unlabeled-id-path dataset/splits/lisc/1_4/split_kmeans/unlabeled.txt \
--pseudo-mask-path outdir/pseudo_masks/$semi_setting \
--save-path outdir/models/$semi_setting #--plus --reliable-id-path outdir/reliable_ids/$semi_setting