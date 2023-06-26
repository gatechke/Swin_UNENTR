#!/usr/bin/env bash

source activate SwinUNETR

python -m torch.distributed.launch --nproc_per_node=2 --master_port=11223 main.py \
  --batch_size=1 \
  --sw_batch_size=2 \
  --num_steps=100000 \
  --lrdecay \
  --eval_num=2000 \
  --logdir='copd_pretrain_288_96' \
  --lr=6e-6 \
  --decay=0.1