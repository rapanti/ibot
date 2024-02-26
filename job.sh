#!/bin/bash 
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH -t 6-00:00:00
#SBATCH --gres=gpu:8
#SBATCH -J ibot-hvp_step3-ep100-seed0
#SBATCH -o logs/%A.%a.%N.txt
#SBATCH -a 0-2%1

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source ~/.profile
conda activate torch

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
  ibot/main_ibot.py \
  --data_path /data/datasets/ImageNet/imagenet-pytorch \
  --batch_size_per_gpu 128 \
  --norm_last_layer false \
  --shared_head true \
  --pred_ratio 0 0.3 \
  --pred_ratio_var 0 0.2 \
  --use_hvp true \
  --hvp_step 3 \
  --local_crops_number 2 \
  --global_crops_number_loader 4 \
  --local_crops_number_loader 4 \
  --epochs 100 \
  --seed 0

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
  ibot/eval_linear.py \
  --data_path /data/datasets/ImageNet/imagenet-pytorch \
  --pretrained_weights checkpoint.pth \
  --load_from checkpoint_teacher_linear.pth
