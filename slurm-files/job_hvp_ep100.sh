#!/bin/bash 
#SBATCH -p ml_gpu-rtxA6000
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:8,localtmp:250
#SBATCH -J ibot-wbs-test
#SBATCH -o %A_%N.txt
# SBATCH -a 0-9%1

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source ~/.profile
conda activate ibot

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
  code/main_ibot.py \
  --data_path /work/dlclarge2/rapanti-hvs/imagenet-wds \
  --load_from checkpoint.pth \
  --epochs 100 \
  --batch_size_per_gpu 32 \
  --global_crops_number_loader 4 \
  --use_hvp true \
  --seed 0

torchrun \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --nnodes=1 \
  --rdzv-endpoint=localhost:0 \
  --rdzv-backend=c10d \
  --rdzv-id=$SLURM_JOB_ID \
  code/eval_linear.py \
    --data_path /work/dlclarge2/rapanti-hvs/imagenet-wds \
    --load_from checkpoint_teacher_linear.pth
