export semi_setting='dataset2/1_4/split_0/deeplabv3plus/resnet50'
python3 -W ignore main.py \
--dataset dataset2 --data-root ./  \
--batch-size 16 --backbone resnet50 --model deeplabv3plus \
--labeled-id-path dataset/splits/dataset2/1_4/split_0/labeled.txt \
--unlabeled-id-path dataset/splits/dataset2/1_4/split_0/unlabeled.txt \
--pseudo-mask-path outdir/pseudo_masks/$semi_setting \
--save-path outdir/models/$semi_setting --plus --reliable-id-path outdir/reliable_ids/$semi_setting