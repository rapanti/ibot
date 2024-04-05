#!/bin/bash
#SBATCH --partition=general
#SBATCH --time=30:00
#SBATCH --nodes=1 #increase it in case of multi-node jobs
#SBATCH --ntasks-per-node=8 #8 maximum possible tasks per node (8 tiles)
#SBATCH --account=pn68xi
#SBATCH --export=NONE
#SBATCH --ear=off
#SBATCH -J ibot-test
# SBATCH -o slurm/%A.%a.%N.txt
# SBATCH --array 0-5%1


# load oneapi base and hpc (called intel-toolkit on phase 2)
module load intel-toolkit/2024.0.0   ## use this one for pytorch v2.1
module load anaconda3

source ~/.conda_init
conda activate hvp 

export LOGLEVEL=INFO

# environment variable to use PVC GPU
export USE_AOT_DEVLIST='ats-m150,pvc'

# environment variables to run multi-tile
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_AFFINITY_MASK=0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1  ## for all 4 gpus / 8 tiles


# environment variables for DDP training - this is not needed for oneapi >= 2023.0 !!
#source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
#export LD_PRELOAD=$(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/lib/libmpi.so
export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
#export CCL_LOG_LEVEL=warn

# environment variables for slurm
export NP=${SLURM_NTASKS}
export NNODES=${SLURM_NNODES}
export PPN=${SLURM_NTASKS_PER_NODE:-$(( NP / NNODES ))}
export J=${SLURM_JOB_ID}
echo "NP =" $NP " PPN =" $PPN

# set up the master_addr/url for running torch distributed multi-node
export MASTER_ADDR=$(mpirun -n 1 -ppn 1 hostname -I | awk '{print $1}')
echo "MASTER_ADDR =" $MASTER_ADDR
export MASTER_PORT=$(shuf -i 10000-60000 -n 1)

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# run the python script
mpirun -n $NP -ppn $PPN -l python -u main_ibot.py
